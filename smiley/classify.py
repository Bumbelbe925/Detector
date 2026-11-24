"""Clasificación de la expresión a partir de ojos y boca.

Este módulo es matemática pura (sin numpy ni OpenCV) para que el mismo
archivo pueda copiarse tal cual a una OpenMV (MicroPython).

Idea:
  - El eje u va de un ojo al otro.
  - El eje v es perpendicular y apunta hacia la boca.
  - En coordenadas (u, v) la cara queda "derecha" aunque esté rotada.
  - La flecha (sagitta) compara la v del centro de la boca contra la v de
    sus extremos, normalizada por el ancho de la boca:
        flecha > 0  -> el centro baja (lejos de los ojos)  -> sonrisa
        flecha < 0  -> el centro sube (cerca de los ojos)  -> tristeza
"""

import math

UNHARMED = "UNHARMED"  # feliz
STABLE = "STABLE"      # neutra
HARMED = "HARMED"      # triste


def face_frame(eye1, eye2, mouth_center):
    """Devuelve (origen, eje_u, eje_v) del sistema de la cara."""
    ux = eye2[0] - eye1[0]
    uy = eye2[1] - eye1[1]
    d = math.sqrt(ux * ux + uy * uy)
    if d == 0:
        return None
    ux, uy = ux / d, uy / d
    mx = (eye1[0] + eye2[0]) / 2.0
    my = (eye1[1] + eye2[1]) / 2.0
    vx, vy = -uy, ux
    if (mouth_center[0] - mx) * vx + (mouth_center[1] - my) * vy < 0:
        vx, vy = -vx, -vy
    return (mx, my), (ux, uy), (vx, vy)


def mouth_analysis(eye1, eye2, mouth_pts, band=0.2):
    """Analiza la boca y devuelve un dict con la flecha y los puntos usados.

    Claves:
      sagitta: flecha normalizada (float)
      left, right, center: puntos (x, y) en la imagen usados para medirla
      origin: punto medio entre los ojos
      u, v: ejes de la cara (vectores unitarios)
    Devuelve None si no se puede calcular.
    """
    n = len(mouth_pts)
    if n < 5:
        return None
    bx = sum(p[0] for p in mouth_pts) / n
    by = sum(p[1] for p in mouth_pts) / n
    frame = face_frame(eye1, eye2, (bx, by))
    if frame is None:
        return None
    (mx, my), (ux, uy), (vx, vy) = frame

    pts = []
    for x, y in mouth_pts:
        dx, dy = x - mx, y - my
        pts.append((dx * ux + dy * uy, dx * vx + dy * vy, x, y))

    u_min = min(p[0] for p in pts)
    u_max = max(p[0] for p in pts)
    width = u_max - u_min
    if width <= 0:
        return None

    t = band * width
    c = (u_min + u_max) / 2.0

    def band_mean(lo, hi):
        sel = [p for p in pts if lo <= p[0] <= hi]
        if not sel:
            return None
        k = len(sel)
        return (sum(p[1] for p in sel) / k,
                (sum(p[2] for p in sel) / k, sum(p[3] for p in sel) / k))

    left = band_mean(u_min, u_min + t)
    right = band_mean(u_max - t, u_max)
    center = band_mean(c - t / 2.0, c + t / 2.0)
    if left is None or right is None or center is None:
        return None
    sag = (center[0] - (left[0] + right[0]) / 2.0) / width
    return {
        "sagitta": sag,
        "left": left[1],
        "right": right[1],
        "center": center[1],
        "origin": (mx, my),
        "u": (ux, uy),
        "v": (vx, vy),
    }


def label_for(sagitta, threshold=0.07):
    """Convierte la flecha en estado: feliz, triste o neutra."""
    if sagitta > threshold:
        return UNHARMED
    if sagitta < -threshold:
        return HARMED
    return STABLE


def classify(eye1, eye2, mouth_pts, threshold=0.07):
    """Devuelve (etiqueta, flecha), o (None, None) si no se pudo decidir."""
    a = mouth_analysis(eye1, eye2, mouth_pts)
    if a is None:
        return None, None
    return label_for(a["sagitta"], threshold), a["sagitta"]
