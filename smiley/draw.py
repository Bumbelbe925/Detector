"""Dibujo de resultados: aro, ojos, boca y los puntos usados para decidir.

annotate(imagen, detecciones) devuelve la imagen completa anotada y, debajo,
un acercamiento de cada cara con los puntos marcados en grande.
"""

from typing import List

import cv2
import numpy as np

from .pipeline import Detection

# Colores BGR
COLORS = {
    "UNHARMED": (0, 170, 0),
    "STABLE": (0, 170, 220),
    "HARMED": (0, 0, 220),
    None: (150, 150, 150),
}
C_EYE = (255, 0, 255)       # ojos y eje u
C_MOUTH = (0, 140, 255)     # píxeles de la boca
C_KEY = (255, 120, 0)       # extremos y centro de la boca
C_V = (200, 200, 0)         # eje v (hacia la boca)
C_ROI = (200, 120, 0)

NOMBRES = {"UNHARMED": "FELIZ", "STABLE": "NEUTRA", "HARMED": "TRISTE", None: "?"}


def _to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()


def _text(img, text, org, scale, color, t):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), t + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, t, cv2.LINE_AA)


def _draw_one(img, d: Detection, tf, s, t, detailed):
    """Dibuja una detección. tf(x, y) transforma coordenadas; s es la escala."""
    def P(p):
        x, y = tf(p[0], p[1])
        return int(round(x)), int(round(y))

    color = COLORS.get(d.label, COLORS[None])
    r = d.ring
    cv2.circle(img, P((r.cx, r.cy)), int(r.r * s), color, t + 1, cv2.LINE_AA)

    f = d.features
    if f is None:
        return
    if not detailed:
        x, y, w, h = f.roi
        cv2.rectangle(img, P((x, y)), P((x + w, y + h)), C_ROI, t)

    # Píxeles de la boca
    step = max(1, len(f.mouth.points) // 400)
    for px, py in f.mouth.points[::step]:
        cv2.circle(img, P((px, py)), max(1, int(s * 0.6)), C_MOUTH, -1)

    # Ojos y eje u
    er = max(3, int(r.r * s * 0.07))
    cv2.line(img, P(f.eye1), P(f.eye2), C_EYE, t, cv2.LINE_AA)
    for e in (f.eye1, f.eye2):
        cv2.circle(img, P(e), er, C_EYE, -1, cv2.LINE_AA)
        cv2.circle(img, P(e), er, (255, 255, 255), max(1, t // 2), cv2.LINE_AA)

    a = d.analysis
    if a is None:
        return
    # Eje v: del punto medio de los ojos hacia la boca
    o = a["origin"]
    v = a["v"]
    tip = (o[0] + v[0] * r.r * 0.45, o[1] + v[1] * r.r * 0.45)
    cv2.arrowedLine(img, P(o), P(tip), C_V, t, cv2.LINE_AA, tipLength=0.25)

    # Cuerda entre extremos y flecha hasta el centro
    L, C, R = a["left"], a["center"], a["right"]
    mid = ((L[0] + R[0]) / 2, (L[1] + R[1]) / 2)
    cv2.line(img, P(L), P(R), C_KEY, t, cv2.LINE_AA)
    cv2.line(img, P(mid), P(C), C_KEY, t, cv2.LINE_AA)
    kr = max(3, int(r.r * s * 0.06))
    for p in (L, C, R):
        cv2.circle(img, P(p), kr, C_KEY, -1, cv2.LINE_AA)
        cv2.circle(img, P(p), kr, (255, 255, 255), max(1, t // 2), cv2.LINE_AA)

    if detailed:
        fs = 0.45
        _text(img, "ojo", (P(f.eye1)[0] + er + 2, P(f.eye1)[1] - er), fs, C_EYE, 1)
        _text(img, "ojo", (P(f.eye2)[0] + er + 2, P(f.eye2)[1] - er), fs, C_EYE, 1)
        _text(img, "L", (P(L)[0] + kr + 2, P(L)[1] + kr), fs, C_KEY, 1)
        _text(img, "R", (P(R)[0] + kr + 2, P(R)[1] + kr), fs, C_KEY, 1)
        _text(img, "C", (P(C)[0] + kr + 2, P(C)[1] + kr), fs, C_KEY, 1)


def draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Imagen completa con las detecciones dibujadas encima."""
    img = _to_bgr(image)
    t = max(1, int(round(min(img.shape[:2]) / 300)))
    for d in detections:
        _draw_one(img, d, lambda x, y: (x, y), 1.0, t, detailed=False)
        r = d.ring
        text = NOMBRES.get(d.label, "?")
        if d.sagitta is not None:
            text += f" ({d.sagitta:+.2f})"
        scale = max(0.35, min(0.9, r.r / 130))
        org = (int(r.cx - r.r), int(max(15, r.cy - r.r - 8)))
        _text(img, text, org, scale, COLORS.get(d.label, COLORS[None]), t)
    return img


def face_zoom(image: np.ndarray, d: Detection, size: int = 300) -> np.ndarray:
    """Acercamiento de una cara con todos los puntos marcados y etiquetados."""
    img = _to_bgr(image)
    r = d.ring
    half = r.r * 1.25
    s = size / (2 * half)
    x0, y0 = r.cx - half, r.cy - half
    M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    crop = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    _draw_one(crop, d, lambda x, y: ((x - x0) * s, (y - y0) * s), s, 2, detailed=True)

    # Barra de título
    bar = np.full((40, size, 3), 255, np.uint8)
    color = COLORS.get(d.label, COLORS[None])
    title = NOMBRES.get(d.label, "?")
    if d.sagitta is not None:
        title += f"   flecha {d.sagitta:+.3f}"
    cv2.putText(bar, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return np.vstack([bar, crop])


def _legend(width: int) -> np.ndarray:
    items = [("ojos / eje u", C_EYE), ("eje v (hacia la boca)", C_V),
             ("boca", C_MOUTH), ("L, C, R = extremos y centro", C_KEY)]
    rows, row, x = [], [], 8
    for txt, c in items:
        w = 16 + int(len(txt) * 8.5) + 18
        if row and x + w > width:
            rows.append(row)
            row, x = [], 8
        row.append((x, txt, c))
        x += w
    rows.append(row)
    leg = np.full((26 * len(rows) + 6, width, 3), 255, np.uint8)
    for i, r in enumerate(rows):
        y = 16 + 26 * i
        for x, txt, c in r:
            cv2.circle(leg, (x + 6, y), 6, c, -1, cv2.LINE_AA)
            cv2.putText(leg, txt, (x + 16, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    return leg


def annotate(image: np.ndarray, detections: List[Detection], zoom_size: int = 300) -> np.ndarray:
    """Imagen completa anotada + fila de acercamientos con los puntos + leyenda."""
    full = draw_detections(image, detections)
    if not detections:
        return full
    zooms = [face_zoom(image, d, zoom_size) for d in detections]
    row = np.hstack(zooms)
    W = max(full.shape[1], row.shape[1])

    def pad(im):
        if im.shape[1] == W:
            return im
        p = np.full((im.shape[0], W - im.shape[1], 3), 255, np.uint8)
        return np.hstack([im, p])

    sep = np.full((6, W, 3), 220, np.uint8)
    return np.vstack([pad(full), sep, pad(row), pad(_legend(W))])
