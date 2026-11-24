"""Mide la precisión del detector sobre imágenes etiquetadas.

El estado correcto se toma del nombre del archivo o de la carpeta:
  - nombre que empiece con UNHARMED_, STABLE_ o HARMED_  (o feliz_, neutra_, triste_)
  - o imágenes dentro de carpetas llamadas así

Uso:
    python tools/evaluate.py sinteticas/
    python tools/evaluate.py fotos_reales/ --errors errores/
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smiley import SmileyDetector  # noqa: E402
from smiley.config import IMAGE_EXTS  # noqa: E402
from smiley.draw import draw_detections  # noqa: E402

ALIASES = {
    "unharmed": "UNHARMED", "feliz": "UNHARMED", "happy": "UNHARMED",
    "stable": "STABLE", "neutra": "STABLE", "neutral": "STABLE",
    "harmed": "HARMED", "triste": "HARMED", "sad": "HARMED",
}


def expected_label(path: Path):
    for part in [path.stem.split("_")[0], path.parent.name]:
        key = part.lower()
        if key in ALIASES:
            return ALIASES[key]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--errors", help="carpeta donde guardar las imágenes mal clasificadas, anotadas")
    a = ap.parse_args()

    det = SmileyDetector()
    files = [p for p in sorted(Path(a.folder).rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    err_dir = Path(a.errors) if a.errors else None
    if err_dir:
        err_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    total = 0
    confusion = Counter()
    sag = {"UNHARMED": [], "STABLE": [], "HARMED": []}
    for p in files:
        exp = expected_label(p)
        if exp is None:
            continue
        img = cv2.imread(str(p))
        dets = det.detect(img)
        got = dets[0].label if dets else "NO_DETECTADA"
        if dets and dets[0].sagitta is not None:
            sag[exp].append(dets[0].sagitta)
        total += 1
        confusion[(exp, got)] += 1
        if got == exp:
            ok += 1
        elif err_dir:
            cv2.imwrite(str(err_dir / f"{p.stem}__{got}.png"), draw_detections(img, dets))

    if total == 0:
        print("No se encontraron imágenes con etiqueta en el nombre.")
        return 1
    print(f"Precisión: {ok}/{total} = {100 * ok / total:.1f}%\n")
    print("Esperado -> Obtenido: cantidad")
    for (e, g), c in sorted(confusion.items()):
        print(f"  {e:9s} -> {g:13s}: {c}")
    print("\nRango de la flecha por clase (sirve para elegir el umbral):")
    for k, v in sag.items():
        if v:
            print(f"  {k:9s}: min {min(v):+.3f}  max {max(v):+.3f}  (n={len(v)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
