"""Paso 1: convertir la imagen en una máscara binaria de lo oscuro."""

import cv2
import numpy as np

from .config import Config


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def dark_mask(image: np.ndarray, cfg: Config) -> np.ndarray:
    """Devuelve una máscara uint8 donde 255 = píxel oscuro (tinta negra)."""
    gray = to_gray(image)
    k = cfg.blur_ksize
    if k and k > 1:
        gray = cv2.GaussianBlur(gray, (k | 1, k | 1), 0)

    if cfg.threshold_mode == "adaptive":
        block = cfg.adaptive_block
        if not block:
            block = max(15, min(gray.shape[:2]) // 6)
        block = block | 1
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
            block, cfg.adaptive_c)
    elif cfg.threshold_mode == "fixed":
        _, mask = cv2.threshold(gray, cfg.fixed_threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask
