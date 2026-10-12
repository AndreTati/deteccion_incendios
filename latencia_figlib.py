"""H3 sobre secuencias de cámara fija de FIgLib: latencia de alerta y falsas alarmas en tiempo real.

Las secuencias de FIgLib (HPWREN) son una foto por minuto, y el MP4 que se descarga del sitio pone
cada foto en un frame (Meadowfire: 70 frames a 4 fps). Por eso no sirve el tiempo del video
(0,25 s de video = 1 min real) y acá cada frame lleva su tiempo real, tomado del nombre de la
foto original (`<unix>_<offset>.jpg`, offset en segundos respecto de la ignición informada).
Los clips, sus offsets, el segundo en que el humo se vuelve visible y la región del evento están
en `configs/figlib_clips.json`.

Ojo: N en K-de-N cuenta frames procesados, y acá un frame es un minuto. K/N = 3/5 exige
detecciones en 3 de los últimos 5 minutos.

1. Video → detecciones (umbral bajo) + p_real del clasificador, con t en segundos reales.
2. Barrido de τ × K/N, con y sin clasificador, con la misma lógica de `alerta.py`, de dos formas:
   - Sistema (como corre la demo, todas las detecciones): alertas antes del evento visible
     (`fa_sist`), porcentaje de esas fotos en ALERTA (`pct_alerta`: un falso positivo fijo es una
     sola falsa alarma pero deja al sistema en alerta todo el tiempo), si al llegar el evento ya
     estaba en alerta (`previa`; entonces el incendio no genera una alerta nueva) y la primera
     alerta desde el evento (`lat_sist`).
   - Por región (`roi` del clip): las detecciones fuera de la región del humo son falsos positivos.
     `fa_roi` cuenta las falsas alarmas que generan en todo el clip (más las de la región antes del
     evento) y `lat_roi` es la latencia de alerta solo con las detecciones del humo real. Separa la
     latencia del incendio de un falso positivo fijo que mantiene al sistema en alerta.
   `det_roi` es el primer minuto con una detección confirmada del humo real: lo que agrega el
   criterio temporal es `lat_roi − det_roi`.

Uso (CPU alcanza: son 70–80 frames por clip):
    python latencia_figlib.py                       # todos los clips, R1 a 640
    python latencia_figlib.py --clip meadowfire --detector runs/yolo11n_960_neg_s42/weights/best.pt --imgsz 960
    python latencia_figlib.py --reusar              # recalcula el barrido sin volver a inferir
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

import alerta
import recortes

CLIPS_JSON = Path("configs/figlib_clips.json")
CONF_MIN = 0.05      # umbral bajo del pipeline, igual que inferencia_video.ipynb
IOU_NMS = 0.7
CLF_CONF_MIN = 0.10  # solo se clasifican las cajas que algún τ del barrido puede confirmar
CLF_IMGSZ = 224

TAUS = [0.15, 0.25, 0.40, 0.50, 0.60]
KN = [(1, 1), (2, 2), (2, 3), (3, 5)]
UMBRAL_CLF = 0.50


def inferir(video: Path, offsets: list[int], detector: YOLO, imgsz: int, clf: YOLO | None, device) -> tuple[pd.DataFrame, pd.DataFrame]:
    cap = cv2.VideoCapture(str(video))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_video != len(offsets):
        raise ValueError(f"{video.name}: el video tiene {n_video} frames y la lista de offsets {len(offsets)}")
    idx_real = recortes.indice_clase_real(clf.names) if clf else None

    det_rows, frame_rows = [], []
    for idx, t in enumerate(offsets):
        ok, img = cap.read()
        if not ok:
            raise IOError(f"No se pudo leer el frame {idx}")
        r = detector.predict(img, imgsz=imgsz, conf=CONF_MIN, iou=IOU_NMS, device=device, verbose=False)[0]
        frame_rows.append({"frame": idx, "t": float(t), "n_det": len(r.boxes)})
        filas = [{"frame": idx, "t": float(t), "clase": r.names[int(c)], "conf": cf, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
                 for b, cf, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
        if clf:
            a_clf = [f for f in filas if f["conf"] > CLF_CONF_MIN]
            if a_clf:
                crops = [recortes.recortar(img, f["x1"], f["y1"], f["x2"], f["y2"]) for f in a_clf]
                for f, rc in zip(a_clf, clf.predict(crops, imgsz=CLF_IMGSZ, device=device, verbose=False)):
                    f["p_real"] = float(rc.probs.data[idx_real])
            for f in filas:
                f.setdefault("p_real", np.nan)
        det_rows += filas
    cap.release()
    cols = ["frame", "t", "clase", "conf", "x1", "y1", "x2", "y2"] + (["p_real"] if clf else [])
    return pd.DataFrame(det_rows, columns=cols), pd.DataFrame(frame_rows)


def en_roi(det: pd.DataFrame, roi: list[float] | None) -> pd.Series:
    """¿El centro de la caja cae en la región del evento?"""
    if roi is None:
        return pd.Series(False, index=det.index)
    cx, cy = (det["x1"] + det["x2"]) / 2, (det["y1"] + det["y2"]) / 2
    return (cx >= roi[0]) & (cx <= roi[2]) & (cy >= roi[1]) & (cy <= roi[3])


def primera_desde(eventos: pd.DataFrame, t_evento: float) -> float:
    post = eventos[eventos["t"] >= t_evento]
    return (post["t"].min() - t_evento) / 60 if len(post) else np.nan


def barrer(frames: pd.DataFrame, det: pd.DataFrame, t_evento: float, roi: list[float] | None) -> pd.DataFrame:
    umbrales = [None, UMBRAL_CLF] if "p_real" in det.columns else [None]
    dentro = en_roi(det, roi)
    filas = []
    for tau, (k, n), u in itertools.product(TAUS, KN, umbrales):
        cfg = alerta.ConfigAlerta(tau=tau, k=k, n=n, umbral_clf=u)
        estados, eventos = alerta.simular(frames, det, cfg)
        antes = estados[estados["t"] < t_evento]
        fila = {
            "tau": tau, "KN": f"{k}/{n}", "clf": u is not None,
            "fa_sist": int((eventos["t"] < t_evento).sum()),
            "pct_alerta": 100 * (antes["estado"] == alerta.ALERTA).mean() if len(antes) else np.nan,
            "previa": bool(len(antes)) and antes["estado"].iloc[-1] == alerta.ALERTA,
            "lat_sist": primera_desde(eventos, t_evento),
        }
        if roi is not None:
            _, ev_fuera = alerta.simular(frames, det[~dentro], cfg)
            _, ev_roi = alerta.simular(frames, det[dentro], cfg)
            confirmadas = det[dentro & (det["t"] >= t_evento)]
            confirmadas = confirmadas[confirmadas.apply(lambda d: alerta.confirmada(d, cfg), axis=1)] if len(confirmadas) else confirmadas
            fila |= {
                "fa_roi": len(ev_fuera) + int((ev_roi["t"] < t_evento).sum()),
                "det_roi": (confirmadas["t"].min() - t_evento) / 60 if len(confirmadas) else np.nan,
                "lat_roi": primera_desde(ev_roi, t_evento),
            }
        filas.append(fila)
    return pd.DataFrame(filas)


def linea_de_tiempo(frames: pd.DataFrame, det: pd.DataFrame, t_evento: float, roi: list[float] | None, desde_min: int = 3):
    """Confianza máxima de humo por foto, en la región del evento y fuera de ella."""
    smoke = det[det["clase"] == "smoke"]
    dentro = en_roi(smoke, roi)
    c_roi = smoke[dentro].groupby("t")["conf"].max()
    c_fuera = smoke[~dentro].groupby("t")["conf"].max()
    print(f"\nConfianza máxima de humo por foto ({'región del evento | fuera' if roi else 'toda la imagen'}):")
    for t in frames["t"][frames["t"] >= t_evento - 60 * desde_min]:
        a, b = (c_roi.get(t, 0.0), c_fuera.get(t, 0.0)) if roi else (c_fuera.get(t, 0.0), None)
        marca = " <- evento visible" if t == t_evento else ""
        extra = f" | {b:.2f}" if b is not None else ""
        print(f"  {int(t):+06d} s  {a:.2f} {'#' * int(a * 30):<30}{extra}{marca}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", nargs="*", help="clips de configs/figlib_clips.json (por defecto, todos)")
    ap.add_argument("--detector", type=Path, default=Path("runs/yolo11n_640_neg_s42/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--clasificador", type=Path, default=Path("runs/clasificador/weights/best.pt"))
    ap.add_argument("--sin-clasificador", action="store_true")
    ap.add_argument("--reusar", action="store_true", help="usar los CSV de detecciones ya generados")
    args = ap.parse_args()

    clips = json.loads(CLIPS_JSON.read_text(encoding="utf-8"))
    nombres = args.clip or list(clips)
    device = 0 if torch.cuda.is_available() else "cpu"
    det_tag = args.detector.parent.parent.name
    out_dir = Path("results") / "video" / det_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = clf = None
    pd.set_option("display.width", 200)

    for nombre in nombres:
        c = clips[nombre]
        rutas = {k: out_dir / f"{nombre}_{k}.csv" for k in ("detecciones", "frames")}
        if args.reusar and all(p.exists() for p in rutas.values()):
            det, frames = pd.read_csv(rutas["detecciones"]), pd.read_csv(rutas["frames"])
        else:
            if detector is None:
                detector = YOLO(args.detector)
                clf = None if args.sin_clasificador or not args.clasificador.exists() else YOLO(args.clasificador)
            det, frames = inferir(Path(c["video"]), c["offsets_s"], detector, args.imgsz, clf, device)
            det.to_csv(rutas["detecciones"], index=False)
            frames.to_csv(rutas["frames"], index=False)
            (out_dir / f"{nombre}_meta.json").write_text(json.dumps({
                "video": c["video"], "secuencia": c["secuencia"], "detector": str(args.detector), "imgsz": args.imgsz,
                "clasificador": str(args.clasificador) if clf else None,
                "tiempo": "real, offset FIgLib en segundos (1 frame = 1 foto, ~1 min)",
            }, indent=2, ensure_ascii=False), encoding="utf-8")

        t_evento, roi = c["t_evento_s"], c.get("roi")
        linea_de_tiempo(frames, det, t_evento, roi)
        barrido = barrer(frames, det, t_evento, roi)
        barrido.to_csv(Path("results") / f"alerta_figlib_{nombre}_{det_tag}.csv", index=False)
        print(f"\n{nombre} | {det_tag} @ {args.imgsz} | evento visible en {t_evento:+.0f} s | "
              f"{(frames['t'] < t_evento).sum()} fotos antes, {len(frames)} en total | tiempos en minutos")
        print(barrido.round({"pct_alerta": 0, "lat_sist": 1, "det_roi": 1, "lat_roi": 1}).to_string(index=False))


if __name__ == "__main__":
    main()
