"""Recorte de cajas para el clasificador liviano.

El clasificador tiene que ver exactamente el mismo tipo de recorte al entrenarse
(clasificador.ipynb, a partir de cajas de ground truth y de falsos positivos del detector)
y al usarse (inferencia_video.ipynb, demo.py). Por eso el recorte vive en un solo lugar.
"""

from __future__ import annotations

import numpy as np

MARGEN = 0.10  # fracción del ancho/alto de la caja que se agrega de contexto por lado
CLASE_REAL = "real"  # nombre de la clase positiva del clasificador (la otra es "confundible")


def recortar(img: np.ndarray, x1: float, y1: float, x2: float, y2: float, margen: float = MARGEN) -> np.ndarray:
    """Recorta la caja (en píxeles, xyxy) con `margen` de contexto, sin salirse de la imagen."""
    h, w = img.shape[:2]
    mx, my = (x2 - x1) * margen, (y2 - y1) * margen
    xa, ya = max(0, int(round(x1 - mx))), max(0, int(round(y1 - my)))
    xb, yb = min(w, int(round(x2 + mx))), min(h, int(round(y2 + my)))
    if xb <= xa or yb <= ya:
        raise ValueError(f"Caja vacía tras recortar: {(x1, y1, x2, y2)} en imagen {w}x{h}")
    return img[ya:yb, xa:xb]


def indice_clase_real(names: dict[int, str]) -> int:
    """Índice de la clase `real` en un modelo de clasificación de Ultralytics."""
    idx = [i for i, n in names.items() if n == CLASE_REAL]
    if not idx:
        raise ValueError(f"El clasificador no tiene clase '{CLASE_REAL}': {names}")
    return idx[0]
