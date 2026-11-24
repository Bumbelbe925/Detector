"""Paso 3: dentro del ROI de cada aro, encontrar ojos y boca."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .ring import Ring


@dataclass
class Blob:
    cx: float
    cy: float
    area: int
    w: int
    h: int
    points: np.ndarray  # (N, 2) en coordenadas de la imagen completa (x, y)

    @property
    def length(self) -> float:
        """Tamaño mayor aproximado (sirve para distinguir boca de ojos)."""
        if len(self.points) < 2:
            return 0.0
        rect = cv2.minAreaRect(self.points.astype(np.float32))
        return float(max(rect[1]))


@dataclass
class FaceFeatures:
    eye1: Tuple[float, float]
    eye2: Tuple[float, float]
    mouth: Blob
    eyes: List[Blob]
    roi: Tuple[int, int, int, int]  # (x, y, w, h)


def roi_for_ring(ring: Ring, shape) -> Tuple[int, int, int, int]:
    """Cuadrado que encierra el interior del aro (recortado a la imagen)."""
    H, W = shape[:2]
    half = ring.r * 0.95
    x0 = int(max(0, ring.cx - half))
    y0 = int(max(0, ring.cy - half))
    x1 = int(min(W, ring.cx + half))
    y1 = int(min(H, ring.cy + half))
    return x0, y0, x1 - x0, y1 - y0


def interior_mask(ring: Ring, x: int, y: int, w: int, h: int, cfg: Config) -> np.ndarray:
    """255 dentro del interior del aro (elipse escalada), 0 en el aro y afuera."""
    margin = cfg.ring_cut_margin * ring.inner_frac
    m = np.zeros((h, w), np.uint8)
    if ring.ellipse is not None:
        (ex, ey), (a1, a2), ang = ring.ellipse
        axes = (max(1, int(a1 / 2 * margin)), max(1, int(a2 / 2 * margin)))
        cv2.ellipse(m, (int(round(ex - x)), int(round(ey - y))), axes, ang, 0, 360, 255, -1)
    else:
        cv2.circle(m, (int(round(ring.cx - x)), int(round(ring.cy - y))), max(1, int(ring.r * margin)), 255, -1)
    return m


def inner_blobs(mask: np.ndarray, ring: Ring, cfg: Config) -> Tuple[List[Blob], tuple]:
    """Blobs oscuros dentro del aro, excluyendo el propio aro."""
    x, y, w, h = roi_for_ring(ring, mask.shape)
    sub = mask[y:y + h, x:x + w].copy()
    # Quita el aro por geometría: todo lo que esté fuera del borde interior
    # (con margen). Así, si la boca toca el aro, la parte interior de la boca sobrevive.
    sub[interior_mask(ring, x, y, w, h, cfg) == 0] = 0

    n, labels, stats, cents = cv2.connectedComponentsWithStats(sub, connectivity=8)
    min_area = cfg.min_feature_area_frac * np.pi * ring.r ** 2
    max_dist = cfg.inner_radius_frac * ring.r
    blobs: List[Blob] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = cents[i][0] + x, cents[i][1] + y
        if np.hypot(cx - ring.cx, cy - ring.cy) > max_dist:
            continue
        ys, xs = np.nonzero(labels == i)
        pts = np.stack([xs + x, ys + y], axis=1)
        blobs.append(Blob(float(cx), float(cy), area,
                          int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]), pts))
    return blobs, (x, y, w, h)


def pick_eyes_and_mouth(blobs: List[Blob], ring: Ring, cfg: Config) -> Optional[Tuple[List[Blob], Blob]]:
    """Elige 2 ojos y 1 boca. Devuelve None si no hay una combinación válida."""
    if len(blobs) < 3:
        return None

    best = None
    # Pocas combinaciones: probamos cada blob como boca y los 2 restantes más parecidos como ojos.
    for mi, mouth in enumerate(blobs):
        others = [b for j, b in enumerate(blobs) if j != mi]
        for a in range(len(others)):
            for b in range(a + 1, len(others)):
                e1, e2 = others[a], others[b]
                big, small = max(e1.area, e2.area), max(1, min(e1.area, e2.area))
                if big / small > cfg.max_eye_area_ratio:
                    continue
                d = np.hypot(e1.cx - e2.cx, e1.cy - e2.cy)
                if not (cfg.min_eye_dist_frac * ring.r <= d <= cfg.max_eye_dist_frac * ring.r):
                    continue
                # La boca debe ser más larga que cualquiera de los ojos.
                if mouth.length <= max(e1.length, e2.length):
                    continue
                # El centro de la cara debe quedar entre los ojos y la boca.
                mx, my = (e1.cx + e2.cx) / 2, (e1.cy + e2.cy) / 2
                v_mouth = np.array([mouth.cx - mx, mouth.cy - my])
                v_center = np.array([ring.cx - mx, ring.cy - my])
                if np.dot(v_mouth, v_center) <= 0:
                    continue
                # Puntuación: ojos parecidos y boca larga.
                score = (big / small) - mouth.length / ring.r
                if best is None or score < best[0]:
                    best = (score, [e1, e2], mouth)
    if best is None:
        return None
    return best[1], best[2]


def extract_features(mask: np.ndarray, ring: Ring, cfg: Config) -> Optional[FaceFeatures]:
    """Ojos y boca de un aro, o None si no parece una cara."""
    blobs, roi = inner_blobs(mask, ring, cfg)
    picked = pick_eyes_and_mouth(blobs, ring, cfg)
    if picked is None:
        return None
    eyes, mouth = picked
    return FaceFeatures((eyes[0].cx, eyes[0].cy), (eyes[1].cx, eyes[1].cy), mouth, eyes, roi)
