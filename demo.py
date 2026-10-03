"""Demo del sistema: video → detección → clasificación de recortes → criterio K-de-N → HUD → MP4.

Es la etapa de inferencia del pipeline completa, frame a frame, con la misma lógica de alerta
(`alerta.py`) y el mismo recorte (`recortes.py`) que los notebooks.

HUD:
- Cajas con clase y confianza (fuego en rojo, humo en celeste). Las que el clasificador de
  recortes descarta se dibujan punteadas en gris.
- Contador del criterio por clase (por ejemplo `humo 3/5`) con barra; la marca indica K.
- Estado del sistema NORMAL / PRE-ALERTA / ALERTA en el borde del video.
- ms/frame y FPS del sistema (detector + clasificador + alerta), promediados (EMA).
- Log de eventos con timestamp del video.

Uso:
    python demo.py --video videos/humo.mp4 --salida demo_humo.mp4 \
        --detector runs/yolo11n_640_neg_s42/weights/best.pt \
        --clasificador runs/clasificador/weights/best.pt --tau 0.4 --k 3 --n 5
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import alerta
import recortes

NOMBRES = {"fire": "fuego", "smoke": "humo"}
COLORES = {"fire": (40, 40, 230), "smoke": (230, 180, 60)}  # BGR
COLOR_DESCARTADA = (150, 150, 150)
COLOR_ESTADO = {alerta.NORMAL: (80, 180, 60), alerta.PRE_ALERTA: (0, 180, 255), alerta.ALERTA: (0, 0, 230)}
FUENTE = cv2.FONT_HERSHEY_SIMPLEX


class Sistema:
    """Detector + clasificador de recortes + criterio temporal, frame a frame."""

    def __init__(self, detector: YOLO, cfg: alerta.ConfigAlerta, clasificador: YOLO | None = None,
                 imgsz: int = 640, conf_min: float = 0.05, clf_imgsz: int = 224, device=None):
        if cfg.umbral_clf is not None and clasificador is None:
            raise ValueError("umbral_clf definido sin clasificador")
        self.detector, self.clasificador, self.cfg = detector, clasificador, cfg
        self.imgsz, self.conf_min, self.clf_imgsz = imgsz, conf_min, clf_imgsz
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.alerta = alerta.AlertaKdeN(cfg)
        self.idx_real = recortes.indice_clase_real(clasificador.names) if clasificador else None

    def procesar(self, frame: int, t: float, img: np.ndarray) -> dict:
        t0 = time.perf_counter()
        r = self.detector.predict(img, imgsz=self.imgsz, conf=self.conf_min, device=self.device, verbose=False)[0]
        dets = [
            {"clase": r.names[int(c)], "conf": conf, "xyxy": xyxy}
            for xyxy, conf, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
        ]
        # Solo se clasifican las cajas que podrían contar para la alerta (conf > τ)
        candidatas = [d for d in dets if d["conf"] > self.cfg.tau and d["clase"] in self.cfg.clases]
        if self.clasificador is not None and candidatas:
            crops = [recortes.recortar(img, *d["xyxy"]) for d in candidatas]
            for d, p in zip(candidatas, self.clasificador.predict(crops, imgsz=self.clf_imgsz,
                                                                 device=self.device, verbose=False)):
                d["p_real"] = float(p.probs.data[self.idx_real])
        for d in candidatas:
            d["confirmada"] = alerta.confirmada(d, self.cfg)
        estado = self.alerta.actualizar(frame, t, dets)
        return {"detecciones": candidatas, "estado": estado, "ms": (time.perf_counter() - t0) * 1000}


def formato_t(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def texto_evento(ev: dict) -> str:
    conf = f"{ev['conf']:.2f}".replace(".", ",")
    return f"{formato_t(ev['t'])} - ALERTA {NOMBRES.get(ev['clase'], ev['clase'])} (conf {conf} - {ev['hits']}/{ev['n']} frames)"


def rect_punteado(img, p1, p2, color, grosor=1, paso=8):
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, 2 * paso):
        for y in (y1, y2):
            cv2.line(img, (x, y), (min(x + paso, x2), y), color, grosor)
    for y in range(y1, y2, 2 * paso):
        for x in (x1, x2):
            cv2.line(img, (x, y), (x, min(y + paso, y2)), color, grosor)


def texto(img, s, org, escala=0.55, color=(255, 255, 255), grosor=1):
    # Contorno negro con el mismo grosor desplazado 1 px (en OpenCV el ancho de las letras
    # depende del grosor, así que un contorno más grueso se desfasa en textos largos)
    x, y = org
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(img, s, (x + dx, y + dy), FUENTE, escala, (0, 0, 0), grosor, cv2.LINE_AA)
    cv2.putText(img, s, org, FUENTE, escala, color, grosor, cv2.LINE_AA)


def dibujar_hud(img: np.ndarray, res: dict, fps: float, log: list[str], cfg: alerta.ConfigAlerta) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    for d in res["detecciones"]:
        x1, y1, x2, y2 = (int(round(v)) for v in d["xyxy"])
        nombre = NOMBRES.get(d["clase"], d["clase"])
        if d["confirmada"]:
            color = COLORES.get(d["clase"], (255, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            etiqueta = f"{nombre} {d['conf']:.2f}" + (f" | real {d['p_real']:.2f}" if "p_real" in d else "")
        else:
            color = COLOR_DESCARTADA
            rect_punteado(out, (x1, y1), (x2, y2), color, 1)
            etiqueta = f"{nombre} {d['conf']:.2f} | descartada {d['p_real']:.2f}"
        texto(out, etiqueta, (x1, max(15, y1 - 6)), 0.45, color)

    # Panel de estado
    estado = res["estado"]
    color_estado = COLOR_ESTADO[estado["estado"]]
    panel_h = 58 + 26 * len(cfg.clases)
    capa = out.copy()
    cv2.rectangle(capa, (10, 10), (330, 10 + panel_h), (0, 0, 0), -1)
    out = cv2.addWeighted(capa, 0.55, out, 0.45, 0)
    texto(out, estado["estado"], (20, 38), 0.8, color_estado, 2)
    y = 66
    for c in cfg.clases:
        v = estado["por_clase"][c]
        texto(out, f"{NOMBRES.get(c, c):<6} {v['hits']}/{v['n']}", (20, y), 0.5, COLORES.get(c, (255, 255, 255)))
        x0, x1b = 130, 310
        cv2.rectangle(out, (x0, y - 11), (x1b, y + 1), (90, 90, 90), 1)
        lleno = x0 + int((x1b - x0) * v["hits"] / v["n"])
        cv2.rectangle(out, (x0, y - 11), (lleno, y + 1), COLOR_ESTADO[v["estado"]], -1)
        xk = x0 + int((x1b - x0) * cfg.k / cfg.n)
        cv2.line(out, (xk, y - 15), (xk, y + 5), (255, 255, 255), 1)
        y += 26
    texto(out, f"{1000 / fps:.0f} ms/frame | {fps:.1f} FPS | tau {cfg.tau:.2f} K/N {cfg.k}/{cfg.n}"
          + (f" | clf {cfg.umbral_clf:.2f}" if cfg.umbral_clf is not None else ""), (20, y), 0.42)

    # Log de eventos (los últimos 4)
    for i, linea in enumerate(log[-4:]):
        texto(out, linea, (14, h - 14 - 22 * (min(len(log), 4) - 1 - i)), 0.5, (0, 0, 255) if i == len(log[-4:]) - 1 else (255, 255, 255))

    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color_estado, 8 if estado["estado"] != alerta.NORMAL else 4)
    return out


def procesar_video(video: Path, salida: Path, sistema: Sistema, vid_stride: int = 1,
                   mostrar: bool = False, log_csv: Path | None = None) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir {video}")
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    salida.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), fps_video / vid_stride, (w, h))

    log, eventos, ms_ema, idx = [], [], None, -1
    while True:
        idx += 1
        if idx % vid_stride:
            if not cap.grab():
                break
            continue
        ok, img = cap.read()
        if not ok:
            break
        res = sistema.procesar(idx, idx / fps_video, img)
        ms_ema = res["ms"] if ms_ema is None else 0.9 * ms_ema + 0.1 * res["ms"]
        for ev in res["estado"]["eventos"]:
            eventos.append(ev)
            log.append(texto_evento(ev))
            print(log[-1])
        frame = dibujar_hud(img, res, 1000 / ms_ema, log, sistema.cfg)
        writer.write(frame)
        if mostrar:
            cv2.imshow("demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    writer.release()
    if mostrar:
        cv2.destroyAllWindows()

    if log_csv is not None:
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["frame", "t", "clase", "conf", "hits", "n"])
            wr.writeheader()
            wr.writerows(eventos)
    return eventos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True, help="MP4 anotado")
    ap.add_argument("--detector", type=Path, required=True, help="best.pt del detector")
    ap.add_argument("--clasificador", type=Path, help="best.pt del clasificador de recortes (opcional)")
    ap.add_argument("--imgsz", type=int, default=640, help="el mismo imgsz con que se entrenó el detector")
    ap.add_argument("--conf-min", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.40)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--umbral-clf", type=float, default=0.50)
    ap.add_argument("--clases", default="smoke,fire")
    ap.add_argument("--vid-stride", type=int, default=1)
    ap.add_argument("--mostrar", action="store_true", help="ver el video mientras se procesa (q para salir)")
    ap.add_argument("--log-csv", type=Path, help="CSV con los eventos de alerta")
    args = ap.parse_args()

    clasificador = YOLO(args.clasificador) if args.clasificador else None
    cfg = alerta.ConfigAlerta(tau=args.tau, k=args.k, n=args.n, clases=tuple(args.clases.split(",")),
                              umbral_clf=args.umbral_clf if clasificador else None)
    sistema = Sistema(YOLO(args.detector), cfg, clasificador, imgsz=args.imgsz, conf_min=args.conf_min)
    eventos = procesar_video(args.video, args.salida, sistema, args.vid_stride, args.mostrar, args.log_csv)
    print(f"{len(eventos)} alertas | video anotado: {args.salida}")


if __name__ == "__main__":
    main()
