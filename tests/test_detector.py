"""Pruebas automáticas. Correr con:  python -m unittest discover tests -v"""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import generate_synthetic  # noqa: E402

from smiley import HARMED, STABLE, UNHARMED, SmileyDetector, classify  # noqa: E402

EJEMPLO = ROOT / "ejemplos" / "tres_caras.png"


def crop_faces(img):
    """Recorta las 3 caras de la imagen de ejemplo (izq, centro, der)."""
    h, w = img.shape[:2]
    third = w // 3
    return [img[:, i * third:(i + 1) * third] for i in range(3)]


def rotate_padded(img, angle):
    h, w = img.shape[:2]
    s = int(max(h, w) * 1.5)
    canvas = np.full((s, s, 3), 255, np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    M = cv2.getRotationMatrix2D((s / 2, s / 2), angle, 1.0)
    return cv2.warpAffine(canvas, M, (s, s), borderValue=(255, 255, 255))


class TestClassifyMath(unittest.TestCase):
    """La función pura, sin imágenes."""

    def arc(self, sign, angle_deg=0):
        pts = []
        for i in range(-20, 21):
            x = i
            y = 30 + sign * 8 * (1 - (i / 20) ** 2)  # y crece hacia abajo
            pts.append((x, y))
        eyes = [(-12, 0), (12, 0)]
        a = math.radians(angle_deg)

        def rot(p):
            return (p[0] * math.cos(a) - p[1] * math.sin(a), p[0] * math.sin(a) + p[1] * math.cos(a))
        return rot(eyes[0]), rot(eyes[1]), [rot(p) for p in pts]

    def test_all_rotations(self):
        for ang in range(0, 360, 10):
            for sign, expected in [(1, UNHARMED), (-1, HARMED), (0, STABLE)]:
                e1, e2, m = self.arc(sign, ang)
                label, _ = classify(e1, e2, m)
                self.assertEqual(label, expected, f"ángulo {ang}, esperado {expected}")
                # el orden de los ojos no debe importar
                label2, _ = classify(e2, e1, m)
                self.assertEqual(label2, expected)


@unittest.skipUnless(EJEMPLO.exists(), "falta ejemplos/tres_caras.png")
class TestEjemplo(unittest.TestCase):
    def setUp(self):
        self.det = SmileyDetector()
        self.img = cv2.imread(str(EJEMPLO))

    def test_tres_caras(self):
        labels = [d.label for d in self.det.detect(self.img)]
        self.assertEqual(labels, [HARMED, STABLE, UNHARMED])

    def test_rotadas(self):
        expected = [HARMED, STABLE, UNHARMED]
        for face, exp in zip(crop_faces(self.img), expected):
            for ang in range(0, 360, 15):
                dets = self.det.detect(rotate_padded(face, ang))
                self.assertEqual(len(dets), 1, f"{exp} a {ang}°: {len(dets)} detecciones")
                self.assertEqual(dets[0].label, exp, f"{exp} a {ang}°")


class TestSinteticas(unittest.TestCase):
    def test_precision(self):
        det = SmileyDetector()
        with tempfile.TemporaryDirectory() as tmp:
            for hard in (False, True):
                folder = generate_synthetic.generate(Path(tmp) / str(hard), 150, hard=hard, seed=7)
                ok = 0
                files = sorted(folder.glob("*.png"))
                for f in files:
                    dets = det.detect(cv2.imread(str(f)))
                    if dets and dets[0].label == f.stem.split("_")[0]:
                        ok += 1
                self.assertGreaterEqual(ok / len(files), 0.95, f"hard={hard}: {ok}/{len(files)}")


@unittest.skipUnless(EJEMPLO.exists(), "falta ejemplos/tres_caras.png")
class TestCasosDificiles(unittest.TestCase):
    def test_linea_tocando_el_aro(self):
        """Una unión de pared que pasa tocando el aro no debe impedir la detección."""
        det = SmileyDetector()
        img = cv2.imread(str(EJEMPLO))
        for face, exp in zip(crop_faces(img), [HARMED, STABLE, UNHARMED]):
            f = face.copy()
            h, w = f.shape[:2]
            dets = det.detect(f)
            r = dets[0].ring
            x = int(r.cx + r.r - 3)
            cv2.line(f, (x, 0), (x, h), (40, 40, 40), 3)
            dets = det.detect(f)
            self.assertEqual([d.label for d in dets], [exp])


class TestFalsosPositivos(unittest.TestCase):
    def setUp(self):
        self.det = SmileyDetector()

    def blank(self):
        return np.full((300, 300, 3), 235, np.uint8)

    def test_pared_vacia(self):
        self.assertEqual(self.det.detect(self.blank()), [])

    def test_cuadro_negro(self):
        img = self.blank()
        cv2.rectangle(img, (80, 80), (220, 220), (20, 20, 20), -1)
        self.assertEqual(self.det.detect(img), [])

    def test_letra_h(self):
        img = self.blank()
        cv2.putText(img, "H", (90, 220), cv2.FONT_HERSHEY_SIMPLEX, 6, (20, 20, 20), 18)
        self.assertEqual(self.det.detect(img), [])

    def test_aro_vacio(self):
        img = self.blank()
        cv2.circle(img, (150, 150), 80, (30, 30, 30), 10)
        self.assertEqual(self.det.detect(img), [])

    def test_aro_con_un_punto(self):
        img = self.blank()
        cv2.circle(img, (150, 150), 80, (30, 30, 30), 10)
        cv2.circle(img, (150, 150), 10, (30, 30, 30), -1)
        self.assertEqual(self.det.detect(img), [])


if __name__ == "__main__":
    unittest.main()
