"""Detector de víctimas tipo smiley para RoboCupJunior Rescue Maze.

    from smiley import SmileyDetector
    det = SmileyDetector()
    resultados = det.detect(imagen_bgr)
"""

from .classify import HARMED, STABLE, UNHARMED, classify, label_for
from .config import Config
from .pipeline import Detection, SmileyDetector

__all__ = ["SmileyDetector", "Detection", "Config", "classify", "label_for", "HARMED", "STABLE", "UNHARMED"]
