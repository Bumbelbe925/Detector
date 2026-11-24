"""Línea de comandos.

Ejemplos:
    python -m smiley foto.jpg                  # guarda en detectadas/ si encuentra caras
    python -m smiley carpeta/ --save salida/   # otra carpeta de salida
    python -m smiley foto.jpg --show           # además la muestra en una ventana
    python -m smiley carpeta/ --no-save        # solo imprime resultados
    python -m smiley carpeta/ --json resultados.json
    python -m smiley --camera 0                # webcam en vivo (guarda al detectar)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from .config import IMAGE_EXTS, Config
from .draw import NOMBRES, annotate
from .pipeline import SmileyDetector


def iter_images(paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in IMAGE_EXTS:
                    yield f
        elif p.suffix.lower() in IMAGE_EXTS:
            yield p
        else:
            print(f"[aviso] se ignora {p}", file=sys.stderr)


def build_config(args) -> Config:
    cfg = Config()
    if args.threshold_mode:
        cfg.threshold_mode = args.threshold_mode
    if args.fixed_threshold is not None:
        cfg.fixed_threshold = args.fixed_threshold
    if args.sagitta is not None:
        cfg.sagitta_threshold = args.sagitta
    if args.min_diameter is not None:
        cfg.min_diameter_px = args.min_diameter
    if args.max_diameter is not None:
        cfg.max_diameter_px = args.max_diameter
    return cfg


def output_name(stem: str, dets) -> str:
    labels = "-".join(NOMBRES.get(d.label, "X") for d in dets) or "SIN_CARAS"
    return f"{stem}_{labels}.jpg"


def run_camera(det: SmileyDetector, index: int, save_dir, every: float):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir la cámara {index}")
    print("q = salir   s = guardar este cuadro")
    last = 0.0
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = det.detect(frame)
        view = annotate(frame, dets)
        cv2.imshow("smiley", view)
        key = cv2.waitKey(1) & 0xFF
        now = time.time()
        if save_dir and ((dets and now - last >= every) or key == ord("s")):
            name = output_name(f"camara_{time.strftime('%Y%m%d_%H%M%S')}_{n:03d}", dets)
            cv2.imwrite(str(save_dir / name), view)
            print(f"guardada {save_dir / name}")
            last, n = now, n + 1
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m smiley", description="Detecta caras smiley y su estado.")
    ap.add_argument("inputs", nargs="*", help="imágenes o carpetas")
    ap.add_argument("--save", metavar="CARPETA", default="detectadas",
                    help="carpeta donde guardar las fotos detectadas con los puntos (por defecto: detectadas/)")
    ap.add_argument("--no-save", action="store_true", help="no guardar imágenes")
    ap.add_argument("--save-all", action="store_true", help="guardar también las fotos sin caras")
    ap.add_argument("--every", type=float, default=1.0, help="con --camera: segundos mínimos entre guardados")
    ap.add_argument("--show", action="store_true", help="muestra cada imagen anotada en una ventana")
    ap.add_argument("--json", metavar="ARCHIVO", help="guarda los resultados en JSON")
    ap.add_argument("--camera", type=int, metavar="N", help="usa la webcam N en vivo")
    ap.add_argument("--all", action="store_true", help="incluye aros donde no se pudo clasificar la cara")
    ap.add_argument("--threshold-mode", choices=["otsu", "adaptive", "fixed"])
    ap.add_argument("--fixed-threshold", type=int)
    ap.add_argument("--sagitta", type=float, help="umbral de la flecha (por defecto 0.07)")
    ap.add_argument("--min-diameter", type=int)
    ap.add_argument("--max-diameter", type=int)
    args = ap.parse_args(argv)

    det = SmileyDetector(build_config(args))

    save_dir = None if args.no_save else Path(args.save)
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    if args.camera is not None:
        run_camera(det, args.camera, save_dir, args.every)
        return 0
    if not args.inputs:
        ap.print_help()
        return 1

    results = {}
    for path in iter_images(args.inputs):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"{path}: no se pudo leer")
            continue
        dets = det.detect(img, include_unclassified=args.all)
        results[str(path)] = [d.to_dict() for d in dets]

        if not dets:
            print(f"{path}: sin caras")
        else:
            resumen = ", ".join(
                f"{NOMBRES.get(d.label, '?')}" + (f" ({d.sagitta:+.2f})" if d.sagitta is not None else "")
                for d in dets)
            print(f"{path}: {resumen}")

        if (save_dir and (dets or args.save_all)) or args.show:
            annotated = annotate(img, dets)
            if save_dir and (dets or args.save_all):
                out = save_dir / output_name(path.stem, dets)
                cv2.imwrite(str(out), annotated)
                print(f"    -> {out}")
            if args.show:
                cv2.imshow(str(path.name), annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
