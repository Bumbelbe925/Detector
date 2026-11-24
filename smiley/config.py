"""Parámetros ajustables del detector.

Todos los umbrales viven aquí para poder calibrar sin tocar la lógica.
"""

from dataclasses import dataclass

# Extensiones de imagen que aceptan la línea de comandos y las herramientas.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Config:
    # --- Preprocesado ---
    blur_ksize: int = 5            # desenfoque gaussiano antes de binarizar (impar)
    threshold_mode: str = "adaptive"  # "adaptive" (recomendado), "otsu" o "fixed"
    fixed_threshold: int = 100     # solo con threshold_mode="fixed" (0-255; menor = más oscuro)
    adaptive_block: int = 0        # ventana para "adaptive" (impar). 0 = automático (~1/6 de la imagen)
    adaptive_c: int = 15           # qué tanto más oscuro que su entorno debe ser un píxel

    # --- Aro (contorno negro de la cara) ---
    min_diameter_px: int = 20      # diámetro mínimo del aro en píxeles
    max_diameter_px: int = 2000    # diámetro máximo del aro en píxeles
    max_aspect: float = 1.35       # ancho/alto permitido (1.0 = círculo perfecto)
    min_fill: float = 0.08         # píxeles del aro / área de su caja
    max_fill: float = 0.45         # un blob relleno da ~0.6-0.8 y se descarta
    min_hole_white: float = 0.55   # fracción blanca exigida en el centro del aro
    min_hole_area: float = 0.25    # área del hueco / área exterior (un arco abierto no tiene hueco)
    min_circularity: float = 0.85  # 4*pi*área/perímetro^2 (círculo = 1.0, cuadrado = 0.785)

    # --- Rasgos dentro del aro ---
    inner_radius_frac: float = 0.80   # solo blobs cuyo centro esté a < frac*r del centro
    ring_cut_margin: float = 0.93     # se ignora todo lo que esté más allá de margin * borde interior del aro
    min_feature_area_frac: float = 0.002  # área mínima de un rasgo relativa a r^2*pi
    max_eye_area_ratio: float = 2.5   # el ojo más grande no puede ser > ratio * el chico
    min_eye_dist_frac: float = 0.35   # distancia entre ojos / radio (mín)
    max_eye_dist_frac: float = 1.40   # distancia entre ojos / radio (máx)

    # --- Clasificación ---
    sagitta_threshold: float = 0.07   # |flecha| mayor a esto => feliz o triste
