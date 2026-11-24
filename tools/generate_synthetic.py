"""Genera imágenes sintéticas de caras smiley para probar el detector.

Cada imagen se guarda como <estado>_<n>.png, así el nombre trae la respuesta
correcta y se puede medir la precisión con tools/evaluate.py.

Uso:
    python tools/generate_synthetic.py --out sinteticas --n 300
    python tools/generate_synthetic.py --out sinteticas --n 300 --hard
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

STATES = ["UNHARMED", "STABLE", "HARMED"]


def draw_face(size, state, ink=40, paper=250, thickness_frac=0.06, rng=random):
    """Dibuja una cara centrada en un lienzo cuadrado (escala de grises)."""
    img = np.full((size, size), paper, np.uint8)
    c = size // 2
    r = int(size * 0.42)
    t = max(2, int(r * thickness_frac * 2))
    cv2.circle(img, (c, c), r, ink, t, cv2.LINE_AA)

    # Ojos (elipses verticales)
    ex = int(r * rng.uniform(0.30, 0.40))
    ey = int(r * rng.uniform(0.25, 0.35))
    ew, eh = int(r * 0.11), int(r * 0.19)
    for sx in (-1, 1):
        cv2.ellipse(img, (c + sx * ex, c - ey), (ew, eh), 0, 0, 360, ink, -1, cv2.LINE_AA)

    # Boca
    mw = int(r * rng.uniform(0.42, 0.55))
    my = c + int(r * rng.uniform(0.28, 0.38))
    mt = max(2, int(r * 0.09))
    if state == "STABLE":
        cv2.line(img, (c - mw, my), (c + mw, my), ink, mt, cv2.LINE_AA)
    else:
        depth = int(r * rng.uniform(0.18, 0.30))
        xs = np.linspace(-mw, mw, 60)
        # parábola: sonrisa = extremos arriba, centro abajo
        sign = 1 if state == "UNHARMED" else -1
        ys = sign * depth * (1 - (xs / mw) ** 2)
        if state == "HARMED":
            ys = ys + depth  # baja el arco para que quede debajo de los ojos
        pts = np.stack([c + xs, my - depth // 2 + ys], axis=1).astype(np.int32)
        cv2.polylines(img, [pts], False, ink, mt, cv2.LINE_AA)
    return img


def place_on_wall(face, out_size, rng, hard=False):
    """Rota, escala y pega la cara sobre un fondo tipo pared, con ruido."""
    wall = rng.randint(200, 245)
    bg = np.full((out_size, out_size), wall, np.uint8)
    # textura suave de la pared
    noise = np.random.normal(0, 4, (out_size, out_size)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    angle = rng.uniform(0, 360)
    scale = rng.uniform(0.35, 0.75) * out_size / face.shape[0]
    fs = face.shape[0]
    M = cv2.getRotationMatrix2D((fs / 2, fs / 2), angle, scale)
    cx = rng.uniform(0.35, 0.65) * out_size
    cy = rng.uniform(0.35, 0.65) * out_size
    M[0, 2] += cx - fs / 2
    M[1, 2] += cy - fs / 2
    warped = cv2.warpAffine(face, M, (out_size, out_size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    mask = cv2.warpAffine(np.full_like(face, 255), M, (out_size, out_size), borderValue=0)
    paper = np.clip(bg.astype(np.int16) + 10, 0, 255)
    img = np.where(mask > 0, np.minimum(warped.astype(np.int16), paper), bg).astype(np.uint8)

    if hard:
        # perspectiva leve (cámara no perpendicular a la pared)
        d = out_size * rng.uniform(0.0, 0.10)
        src = np.float32([[0, 0], [out_size, 0], [out_size, out_size], [0, out_size]])
        dst = np.float32([[rng.uniform(0, d), rng.uniform(0, d)],
                          [out_size - rng.uniform(0, d), rng.uniform(0, d)],
                          [out_size - rng.uniform(0, d), out_size - rng.uniform(0, d)],
                          [rng.uniform(0, d), out_size - rng.uniform(0, d)]])
        P = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, P, (out_size, out_size), borderValue=wall)
        # gradiente de iluminación
        g = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.75, 1.0), out_size)
        img = np.clip(img * (g[None, :] if rng.random() < 0.5 else g[:, None]), 0, 255).astype(np.uint8)
        # desenfoque y ruido de sensor
        k = rng.choice([1, 3, 5])
        if k > 1:
            img = cv2.GaussianBlur(img, (k, k), 0)
        img = np.clip(img.astype(np.int16) + np.random.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    return img


def generate(out_dir: Path, n: int, size: int = 240, hard: bool = False, seed: int = 0):
    rng = random.Random(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        state = STATES[i % 3]
        face = draw_face(400, state, rng=rng)
        img = place_on_wall(face, size, rng, hard=hard)
        cv2.imwrite(str(out_dir / f"{state}_{i:04d}.png"), img)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sinteticas")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--size", type=int, default=240, help="tamaño de la imagen en px")
    ap.add_argument("--hard", action="store_true", help="agrega perspectiva, luz desigual, blur y ruido")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(Path(a.out), a.n, a.size, a.hard, a.seed)
    print(f"{a.n} imágenes en {a.out}/")


if __name__ == "__main__":
    main()
