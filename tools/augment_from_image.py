"""Genera muchas imágenes nuevas a partir de caras reales de una foto.

Toma una o varias imágenes, recorta cada cara que encuentre el detector
(con su estado), y crea variaciones: rotación, tamaño, posición, perspectiva,
luz desigual, color de pared, desenfoque, ruido y compresión JPEG.
Cada archivo se nombra <ESTADO>_<n>.jpg para poder evaluarlo después.

Uso:
    python tools/augment_from_image.py ejemplos/tres_caras.png --out aumentadas --n 300
    python tools/augment_from_image.py fotos/ --out aumentadas --n 600 --size 320x240
    python tools/augment_from_image.py cara.png --label HARMED --out aumentadas
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smiley import SmileyDetector  # noqa: E402
from smiley.config import IMAGE_EXTS  # noqa: E402


def collect_faces(paths, label=None, margin=1.12):
    """Devuelve lista de (recorte_bgr, máscara_circular, estado)."""
    det = SmileyDetector()
    faces = []
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[aviso] no se pudo leer {f}")
            continue
        dets = det.detect(img, include_unclassified=label is not None)
        for d in dets:
            state = label or d.label
            if state is None:
                continue
            r = d.ring.r * margin
            x0, y0 = int(d.ring.cx - r), int(d.ring.cy - r)
            x1, y1 = int(d.ring.cx + r), int(d.ring.cy + r)
            if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
                pad = int(r)
                img_p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
                crop = img_p[y0 + pad:y1 + pad, x0 + pad:x1 + pad].copy()
            else:
                crop = img[y0:y1, x0:x1].copy()
            s = crop.shape[0]
            mask = np.zeros((s, s), np.uint8)
            cv2.circle(mask, (s // 2, s // 2), int(d.ring.r * 1.04 * s / (2 * r)), 255, -1, cv2.LINE_AA)
            faces.append((crop, mask, state))
        print(f"{f}: {len(dets)} cara(s)")
    return faces


def make_wall(w, h, rng):
    """Fondo tipo pared: blanco/gris claro, a veces con tinte, con textura suave."""
    base = rng.randint(185, 245)
    tint = np.array([rng.randint(-8, 8) for _ in range(3)])
    wall = np.clip(np.full((h, w, 3), base, np.int16) + tint, 0, 255)
    noise = np.random.normal(0, rng.uniform(1, 5), (h, w, 1))
    wall = wall + noise
    # a veces una unión de paredes o borde de loseta (distractor)
    if rng.random() < 0.3:
        x = rng.randint(0, w - 1)
        cv2.line(wall, (x, 0), (x + rng.randint(-20, 20), h), (base - 40,) * 3, rng.randint(1, 3))
    return np.clip(wall, 0, 255).astype(np.uint8)


def augment(face, mask, out_w, out_h, rng):
    wall = make_wall(out_w, out_h, rng)
    s = face.shape[0]

    # papel ligeramente distinto a la pared (tono y brillo)
    paper_gain = rng.uniform(0.85, 1.05)
    face_f = np.clip(face.astype(np.float32) * paper_gain, 0, 255)

    # rotación, escala y posición (la cara debe quedar completa en la imagen)
    diameter = rng.uniform(0.25, 0.70) * min(out_w, out_h)
    scale = diameter / s
    angle = rng.uniform(0, 360)
    margin = diameter * 0.6
    cx = rng.uniform(margin, out_w - margin)
    cy = rng.uniform(margin, out_h - margin)
    M = cv2.getRotationMatrix2D((s / 2, s / 2), angle, scale)
    M[0, 2] += cx - s / 2
    M[1, 2] += cy - s / 2
    warped = cv2.warpAffine(face_f, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    m = cv2.warpAffine(mask, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32) / 255
    img = warped * m[..., None] + wall.astype(np.float32) * (1 - m[..., None])

    # perspectiva leve (cámara no perpendicular)
    if rng.random() < 0.7:
        d = min(out_w, out_h) * rng.uniform(0.0, 0.10)
        src = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
        dst = src + np.float32([[rng.uniform(-d, d), rng.uniform(-d, d)] for _ in range(4)])
        P = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, P, (out_w, out_h), borderMode=cv2.BORDER_REPLICATE)

    # iluminación: brillo global + gradiente + mancha de luz del LED
    img *= rng.uniform(0.6, 1.15)
    if rng.random() < 0.7:
        g0, g1 = rng.uniform(0.7, 1.0), rng.uniform(0.7, 1.0)
        if rng.random() < 0.5:
            grad = np.linspace(g0, g1, out_w)[None, :, None]
        else:
            grad = np.linspace(g0, g1, out_h)[:, None, None]
        img *= grad
    if rng.random() < 0.3:
        yy, xx = np.mgrid[0:out_h, 0:out_w]
        sx, sy = rng.uniform(0, out_w), rng.uniform(0, out_h)
        rad = rng.uniform(0.3, 0.7) * max(out_w, out_h)
        spot = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * rad ** 2))
        img += spot[..., None] * rng.uniform(15, 45)

    # balance de blancos
    img *= np.array([rng.uniform(0.92, 1.08) for _ in range(3)])[None, None, :]

    # desenfoque (foco o movimiento)
    r = rng.random()
    if r < 0.3:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    elif r < 0.45:
        k = rng.choice([3, 5, 7])
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        Mk = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        kernel = cv2.warpAffine(kernel, Mk, (k, k))
        kernel /= max(kernel.sum(), 1e-6)
        img = cv2.filter2D(img, -1, kernel)

    # ruido de sensor
    img += np.random.normal(0, rng.uniform(1, 8), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="imágenes o carpetas con caras")
    ap.add_argument("--out", default="aumentadas")
    ap.add_argument("--n", type=int, default=300, help="total de imágenes a generar")
    ap.add_argument("--size", default="320x240", help="ANCHOxALTO de salida (320x240 = QVGA de OpenMV)")
    ap.add_argument("--label", choices=["UNHARMED", "STABLE", "HARMED"],
                    help="forzar el estado (si la foto trae una sola cara conocida)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="guarda también un mosaico _preview.jpg")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    np.random.seed(a.seed)
    out_w, out_h = (int(v) for v in a.size.lower().split("x"))

    faces = collect_faces(a.inputs, a.label)
    if not faces:
        sys.exit("No se encontraron caras en las imágenes de entrada.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    samples = []
    for i in range(a.n):
        face, mask, state = faces[i % len(faces)]
        img = augment(face, mask, out_w, out_h, rng)
        q = rng.randint(60, 95)
        cv2.imwrite(str(out / f"{state}_{i:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, q])
        counts[state] = counts.get(state, 0) + 1
        if len(samples) < 24:
            samples.append(img)

    if a.preview and samples:
        while len(samples) % 6:
            samples.append(np.full_like(samples[0], 255))
        rows = [np.hstack(samples[r:r + 6]) for r in range(0, len(samples), 6)]
        cv2.imwrite(str(out / "_preview.jpg"), np.vstack(rows))

    print(f"\n{a.n} imágenes en {out}/  ->  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
