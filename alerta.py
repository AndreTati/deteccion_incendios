"""Criterio temporal K-de-N y estados de alerta del sistema.

Una clase entra en ALERTA cuando al menos K de los últimos N frames procesados tienen una
detección confirmada de esa clase. Confirmada significa:

- confianza > tau, y
- si se usa el clasificador de recortes, p_real >= umbral_clf.

Estados por clase:

- NORMAL: ninguna detección confirmada en la ventana.
- PRE-ALERTA: entre 1 y K-1 detecciones confirmadas en la ventana.
- ALERTA: K o más. Se emite un evento al entrar.

Histéresis: una vez en ALERTA, la clase sigue en alerta hasta que la ventana queda vacía
(N frames seguidos sin detección confirmada). Así un incendio continuo con detecciones que
parpadean cuenta como un solo evento y no como varias alarmas.

El estado global del sistema es el peor de las clases.

Se usa de dos formas:

- Offline, sobre los CSV de inferencia_video.ipynb (`simular`, `metricas_clip`, `resumir`):
  barrido de tau / K / N con y sin clasificador sin reprocesar el video.
- Online, frame a frame (`AlertaKdeN.actualizar`): demo con HUD.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

NORMAL, PRE_ALERTA, ALERTA = "NORMAL", "PRE-ALERTA", "ALERTA"
NIVEL = {NORMAL: 0, PRE_ALERTA: 1, ALERTA: 2}
CLASES = ("smoke", "fire")


@dataclass(frozen=True)
class ConfigAlerta:
    tau: float = 0.40
    k: int = 3
    n: int = 5
    umbral_clf: float | None = None  # None: sin clasificador de recortes
    clases: tuple[str, ...] = CLASES

    def __post_init__(self):
        if not 1 <= self.k <= self.n:
            raise ValueError(f"Se necesita 1 <= K <= N (K={self.k}, N={self.n})")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError(f"tau fuera de [0, 1]: {self.tau}")

    @property
    def nombre(self) -> str:
        clf = "sin_clf" if self.umbral_clf is None else f"clf{self.umbral_clf:.2f}"
        return f"tau{self.tau:.2f}_k{self.k}n{self.n}_{clf}"


def confirmada(det: Mapping, cfg: ConfigAlerta) -> bool:
    """¿La detección cuenta para el criterio temporal?"""
    if det["conf"] <= cfg.tau:
        return False
    if cfg.umbral_clf is None:
        return True
    p_real = det.get("p_real")
    if p_real is None or pd.isna(p_real):
        raise ValueError("umbral_clf definido pero la detección no tiene p_real: falta pasar el clasificador")
    return p_real >= cfg.umbral_clf


class AlertaKdeN:
    """Buffer deslizante de los últimos N frames por clase, con máquina de estados."""

    def __init__(self, cfg: ConfigAlerta):
        self.cfg = cfg
        # Por frame se guarda la confianza máxima confirmada de la clase, o None si no hubo
        self.buffers = {c: deque(maxlen=cfg.n) for c in cfg.clases}
        self.en_alerta = {c: False for c in cfg.clases}
        self.eventos: list[dict] = []

    def actualizar(self, frame: int, t: float, detecciones: Iterable[Mapping]) -> dict:
        """Procesa un frame. `detecciones`: dicts con 'clase', 'conf' y opcionalmente 'p_real'.

        Devuelve el estado global, el estado de cada clase (con hits/N para el HUD) y los
        eventos emitidos en este frame.
        """
        detecciones = list(detecciones)
        por_clase, nuevos = {}, []
        for c in self.cfg.clases:
            confs = [d["conf"] for d in detecciones if d["clase"] == c and confirmada(d, self.cfg)]
            buf = self.buffers[c]
            buf.append(max(confs) if confs else None)
            hits = sum(x is not None for x in buf)

            if hits >= self.cfg.k and not self.en_alerta[c]:
                self.en_alerta[c] = True
                evento = {
                    "frame": frame, "t": t, "clase": c,
                    "conf": max(x for x in buf if x is not None),
                    "hits": hits, "n": self.cfg.n,
                }
                self.eventos.append(evento)
                nuevos.append(evento)
            elif hits == 0:
                self.en_alerta[c] = False

            estado = ALERTA if self.en_alerta[c] else PRE_ALERTA if hits > 0 else NORMAL
            por_clase[c] = {"estado": estado, "hits": hits, "n": self.cfg.n}

        global_ = max((v["estado"] for v in por_clase.values()), key=NIVEL.get)
        return {"frame": frame, "t": t, "estado": global_, "por_clase": por_clase, "eventos": nuevos}


def simular(frames: pd.DataFrame, detecciones: pd.DataFrame, cfg: ConfigAlerta) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corre la alerta sobre un clip ya inferido.

    frames: una fila por frame procesado (columnas `frame`, `t`), tenga o no detecciones.
    detecciones: una fila por detección (`frame`, `clase`, `conf` y opcionalmente `p_real`).

    Devuelve (estados por frame, eventos).
    """
    cols = ["clase", "conf"] + (["p_real"] if "p_real" in detecciones.columns else [])
    por_frame = {f: g[cols].to_dict("records") for f, g in detecciones.groupby("frame")}

    alerta = AlertaKdeN(cfg)
    filas = []
    for frame, t in frames[["frame", "t"]].itertuples(index=False):
        r = alerta.actualizar(int(frame), float(t), por_frame.get(frame, []))
        fila = {"frame": r["frame"], "t": r["t"], "estado": r["estado"]}
        for c, v in r["por_clase"].items():
            fila[f"hits_{c}"] = v["hits"]
            fila[f"estado_{c}"] = v["estado"]
        filas.append(fila)

    eventos = pd.DataFrame(alerta.eventos, columns=["frame", "t", "clase", "conf", "hits", "n"])
    return pd.DataFrame(filas), eventos


def metricas_clip(eventos: pd.DataFrame, duracion_s: float, t_evento: float | None = None) -> dict:
    """Métricas de alerta de un clip.

    - Clip sin evento (`t_evento=None`): toda alerta es falsa alarma; se observa todo el clip.
    - Clip con evento: las alertas antes de `t_evento` (segundo en que el evento se vuelve
      visible, anotado a mano) son falsas alarmas sobre `t_evento` segundos de observación;
      la latencia es el tiempo desde `t_evento` hasta la primera alerta posterior.
    """
    if t_evento is None or pd.isna(t_evento):
        return {
            "falsas_alarmas": len(eventos), "horas_sin_evento": duracion_s / 3600,
            "detectado": np.nan, "latencia_s": np.nan,
        }
    previas = eventos[eventos["t"] < t_evento]
    posteriores = eventos[eventos["t"] >= t_evento]
    return {
        "falsas_alarmas": len(previas), "horas_sin_evento": t_evento / 3600,
        "detectado": bool(len(posteriores)),
        "latencia_s": float(posteriores["t"].min() - t_evento) if len(posteriores) else np.nan,
    }


def resumir(metricas: pd.DataFrame) -> dict:
    """Agrega `metricas_clip` de varios clips: falsas alarmas/hora, latencia y tasa de detección."""
    horas = metricas["horas_sin_evento"].sum()
    con_evento = metricas["detectado"].notna()
    return {
        "falsas_alarmas": int(metricas["falsas_alarmas"].sum()),
        "falsas_alarmas_por_hora": metricas["falsas_alarmas"].sum() / horas if horas > 0 else np.nan,
        "tasa_eventos_detectados": metricas.loc[con_evento, "detectado"].astype(float).mean(),
        "latencia_media_s": metricas["latencia_s"].mean(),
        "latencia_max_s": metricas["latencia_s"].max(),
    }
