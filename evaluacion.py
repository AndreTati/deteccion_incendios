"""Evaluación del detector a un punto de operación (τ, y opcionalmente umbral del clasificador).

Complementa al mAP de Ultralytics (que integra sobre todos los umbrales) con lo que necesita el
sistema: a un τ fijo, qué predicciones son TP o FP, qué cajas de ground truth quedan sin detectar,
y cuántos falsos positivos hay por imagen negativa. Lo usan clasificador.ipynb (H4) y
analisis_errores.ipynb (peores casos, recall por tamaño para H1, set de estrés).

Convenciones:
- `preds`: una fila por predicción (`imagen, clase, conf, x1, y1, x2, y2` y opcionalmente `p_real`).
- `gts`: una fila por caja de ground truth (`imagen, clase, x1, y1, x2, y2, ancho_img, alto_img`).
- Cajas en píxeles de la imagen original, formato xyxy.
- Match: IoU ≥ 0,5 con una caja de la misma clase, de mayor a menor confianza, cada caja de GT
  se usa una sola vez (como en COCO / Ultralytics).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "smoke", 1: "fire"}
COLS = ["x1", "y1", "x2", "y2"]
SMALL, MEDIUM = 32**2, 96**2  # umbrales COCO de área (px²)


def leer_gt(lbl_path: Path, W: int, H: int) -> list[dict]:
    """Cajas de ground truth de un label YOLO, en píxeles (xyxy) y recortadas al borde de la imagen."""
    cajas = []
    for line in Path(lbl_path).read_text().splitlines():
        if not line.strip():
            continue
        c, x, y, w, h = line.split()[:5]
        x, y, w, h = map(float, (x, y, w, h))
        cajas.append({
            "clase": CLASS_NAMES[int(c)],
            "x1": max(0.0, (x - w / 2) * W), "y1": max(0.0, (y - h / 2) * H),
            "x2": min(float(W), (x + w / 2) * W), "y2": min(float(H), (y + h / 2) * H),
            "ancho_img": W, "alto_img": H,
        })
    return cajas


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre cajas xyxy: a (n, 4), b (m, 4) → (n, m)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def filtrar(preds: pd.DataFrame, tau: float, umbral_clf: float | None = None) -> pd.DataFrame:
    """Predicciones que el sistema reporta al punto de operación."""
    p = preds[preds["conf"] > tau]
    if umbral_clf is not None:
        if p["p_real"].isna().any():
            raise ValueError("Hay predicciones sobre τ sin p_real: falta pasar el clasificador")
        p = p[p["p_real"] >= umbral_clf]
    return p


def emparejar(preds: pd.DataFrame, gts: pd.DataFrame, iou_thr: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Marca cada predicción como TP o FP y cada caja de GT como detectada o no.

    Devuelve copias con columnas nuevas:
    - preds: `tp` (bool), `iou_gt` (IoU con la caja de GT matcheada, o la máxima de su clase si es FP).
    - gts: `detectada` (bool), `conf_det` (confianza de la predicción que la detectó).
    """
    preds = preds.copy()
    gts = gts.copy()
    preds["tp"], preds["iou_gt"] = False, 0.0
    gts["detectada"], gts["conf_det"] = False, np.nan

    gt_idx = {k: g.index.to_numpy() for k, g in gts.groupby(["imagen", "clase"])}
    for (img, clase), p in preds.groupby(["imagen", "clase"]):
        g_idx = gt_idx.get((img, clase))
        if g_idx is None:
            continue
        p = p.sort_values("conf", ascending=False)
        ious = iou(p[COLS].to_numpy(), gts.loc[g_idx, COLS].to_numpy())
        libres = np.ones(len(g_idx), bool)
        for i, pi in enumerate(p.index):
            preds.at[pi, "iou_gt"] = float(ious[i].max())
            cand = np.where(libres & (ious[i] >= iou_thr))[0]
            if len(cand):
                j = cand[np.argmax(ious[i, cand])]
                libres[j] = False
                preds.at[pi, "tp"] = True
                preds.at[pi, "iou_gt"] = float(ious[i, j])
                gts.at[g_idx[j], "detectada"] = True
                gts.at[g_idx[j], "conf_det"] = preds.at[pi, "conf"]
    return preds, gts


def metricas(preds_emp: pd.DataFrame, gts_emp: pd.DataFrame, negativas: list[str],
             clases=tuple(CLASS_NAMES.values())) -> dict:
    """TP, FP, FN, P y R por clase, y falsos positivos por imagen negativa."""
    fila = {}
    for clase in clases:
        p = preds_emp[preds_emp["clase"] == clase]
        g = gts_emp[gts_emp["clase"] == clase]
        tp, fp, n_gt = int(p["tp"].sum()), int((~p["tp"]).sum()), len(g)
        fila |= {f"TP_{clase}": tp, f"FP_{clase}": fp, f"FN_{clase}": n_gt - tp,
                 f"P_{clase}": tp / (tp + fp) if tp + fp else np.nan,
                 f"R_{clase}": tp / n_gt if n_gt else np.nan}
    en_neg = preds_emp["imagen"].isin(negativas)
    n_neg = len(negativas)
    fila["FP_por_img_neg"] = en_neg.sum() / n_neg if n_neg else np.nan
    fila["pct_img_neg_con_FP"] = 100 * preds_emp.loc[en_neg, "imagen"].nunique() / n_neg if n_neg else np.nan
    return fila


def evaluar(preds: pd.DataFrame, gts: pd.DataFrame, negativas: list[str], tau: float,
            umbral_clf: float | None = None) -> dict:
    """Atajo: filtrar → emparejar → metricas."""
    p, g = emparejar(filtrar(preds, tau, umbral_clf), gts)
    return {"tau": tau, "umbral_clf": umbral_clf, **metricas(p, g, negativas)}


def tamano(cajas: pd.DataFrame, imgsz: int) -> pd.Series:
    """Categoría COCO (small / medium / large) del área de cada caja tras el letterbox a `imgsz`."""
    escala = imgsz / cajas[["ancho_img", "alto_img"]].max(axis=1)
    area = (cajas["x2"] - cajas["x1"]) * (cajas["y2"] - cajas["y1"]) * escala**2
    return pd.cut(area, [-np.inf, SMALL, MEDIUM, np.inf], labels=["small", "medium", "large"], right=False)
