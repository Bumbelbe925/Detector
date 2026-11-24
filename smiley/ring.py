"""Paso 2: encontrar los aros negros (contorno de cada cara)."""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .config import Config


@dataclass
class Ring:
    cx: float
    cy: float
    r: float                # radio exterior
    bbox: tuple             # (x, y, w, h)
    label: int              # id del componente en la máscara
    fill: float             # píxeles del aro / área de la caja (0 si vino del respaldo)
    ellipse: Optional[tuple] = None  # ((cx, cy), (eje1, eje2), ángulo) del borde exterior
    inner_frac: float = 0.8  # radio interior / radio exterior (grosor del aro)


def find_rings(mask: np.ndarray, cfg: Config) -> List[Ring]:
    """Devuelve los aros encontrados en la máscara oscura."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rings: List[Ring] = []
    for i in range(1, n):
        ring = _ring_from_component(mask, labels, stats, i, cfg)
        if ring is not None:
            rings.append(ring)
        elif stats[i][2] >= cfg.min_diameter_px and stats[i][3] >= cfg.min_diameter_px:
            # Respaldo: el aro puede estar pegado a una línea o sombra.
            rings.extend(_rings_from_holes(labels, stats, i, cfg))
    return _drop_nested(rings)


def _ring_from_component(mask, labels, stats, i, cfg: Config) -> Optional[Ring]:
    """Intenta interpretar el componente i completo como un aro."""
    x, y, w, h, area = stats[i]
    if w < cfg.min_diameter_px or h < cfg.min_diameter_px:
        return None
    if w > cfg.max_diameter_px or h > cfg.max_diameter_px:
        return None
    if max(w, h) / max(1, min(w, h)) > cfg.max_aspect:
        return None
    fill = area / float(w * h)
    if not (cfg.min_fill <= fill <= cfg.max_fill):
        return None

    cx, cy = x + w / 2.0, y + h / 2.0
    r = (w + h) / 4.0

    # El centro del aro debe ser mayormente blanco (es un aro, no un manchón).
    hr = max(1, int(r * 0.25))
    patch = mask[max(0, int(cy - hr)):int(cy + hr), max(0, int(cx - hr)):int(cx + hr)]
    if patch.size == 0:
        return None
    if 1.0 - np.count_nonzero(patch) / patch.size < cfg.min_hole_white:
        return None

    # Un aro está cerrado: tiene un hueco grande. Una boca en arco no.
    comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
    cnts, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    outer = [c for c, hh in zip(cnts, hier[0]) if hh[3] == -1]
    holes = [c for c, hh in zip(cnts, hier[0]) if hh[3] != -1]
    c = max(outer, key=cv2.contourArea)
    outer_a = cv2.contourArea(c)
    hole_a = max((cv2.contourArea(hc) for hc in holes), default=0.0)
    if outer_a <= 0 or hole_a / outer_a < cfg.min_hole_area:
        return None
    if _circularity(c) < cfg.min_circularity:
        return None

    # Elipse del borde exterior (tolera perspectiva).
    ellipse = None
    if len(c) >= 5:
        (ex, ey), (a1, a2), ang = cv2.fitEllipse(c)
        ellipse = ((ex + x, ey + y), (a1, a2), ang)
        cx, cy = ex + x, ey + y
        r = (a1 + a2) / 4.0

    # Grosor del aro a partir de su área.
    outer_area = np.pi * r * r
    inner_frac = float(np.sqrt(max(0.0, 1.0 - area / outer_area))) if outer_area > 0 else 0.8
    inner_frac = min(max(inner_frac, 0.5), 0.97)

    return Ring(cx, cy, r, (int(x), int(y), int(w), int(h)), i, fill, ellipse, inner_frac)


def _rings_from_holes(labels, stats, comp_id, cfg: Config) -> List[Ring]:
    """Busca huecos redondos (el interior blanco de una cara) dentro de un componente.

    Sirve cuando el aro está pegado a otra cosa (una unión de pared, una sombra)
    y el componente completo ya no parece un círculo.
    """
    x, y, w, h, _ = stats[comp_id]
    comp = (labels[y:y + h, x:x + w] == comp_id).astype(np.uint8)
    cnts, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out: List[Ring] = []
    if not cnts:
        return out
    for c, hh in zip(cnts, hier[0]):
        if hh[3] == -1 or len(c) < 5:
            continue  # solo huecos (contornos interiores)
        if _circularity(c) < cfg.min_circularity:  # el hueco debe ser redondo
            continue
        (ex, ey), (a1, a2), ang = cv2.fitEllipse(c)
        if max(a1, a2) / max(1e-6, min(a1, a2)) > cfg.max_aspect:
            continue
        inner_frac = 0.85  # grosor típico del aro
        d_outer = (a1 + a2) / 2 / inner_frac
        if not (cfg.min_diameter_px <= d_outer <= cfg.max_diameter_px):
            continue
        cx, cy = ex + x, ey + y
        r = d_outer / 2
        ell = ((cx, cy), (a1 / inner_frac, a2 / inner_frac), ang)
        bbox = (int(cx - r), int(cy - r), int(2 * r), int(2 * r))
        out.append(Ring(cx, cy, r, bbox, comp_id, 0.0, ell, inner_frac))
    return out


def _circularity(contour) -> float:
    area = cv2.contourArea(contour)
    per = cv2.arcLength(contour, True)
    return 0.0 if per <= 0 else 4 * np.pi * area / (per * per)


def _drop_nested(rings: List[Ring]) -> List[Ring]:
    """Si un aro está dentro de otro, se queda solo el de afuera."""
    rings.sort(key=lambda rg: -rg.r)
    kept: List[Ring] = []
    for rg in rings:
        if any(np.hypot(rg.cx - k.cx, rg.cy - k.cy) + rg.r <= k.r * 1.05 for k in kept):
            continue
        kept.append(rg)
    return kept
