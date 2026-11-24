"""Une los pasos: imagen -> lista de caras detectadas con su estado."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .classify import label_for, mouth_analysis
from .config import Config
from .features import FaceFeatures, extract_features
from .preprocess import dark_mask
from .ring import Ring, find_rings


@dataclass
class Detection:
    label: Optional[str]          # "UNHARMED", "STABLE", "HARMED" o None
    sagitta: Optional[float]      # valor de la flecha (útil para calibrar)
    ring: Ring
    features: Optional[FaceFeatures]
    analysis: Optional[dict] = None  # puntos usados para decidir (ver classify.mouth_analysis)

    def to_dict(self):
        return {
            "label": self.label,
            "sagitta": None if self.sagitta is None else round(self.sagitta, 4),
            "center": [round(self.ring.cx, 1), round(self.ring.cy, 1)],
            "radius": round(self.ring.r, 1),
            "eyes": None if self.features is None else
                [[round(v, 1) for v in self.features.eye1], [round(v, 1) for v in self.features.eye2]],
            "mouth_points": None if self.analysis is None else
                {k: [round(v, 1) for v in self.analysis[k]] for k in ("left", "center", "right")},
        }


class SmileyDetector:
    """Uso:
        det = SmileyDetector()
        for d in det.detect(imagen_bgr):
            print(d.label)
    """

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or Config()

    def detect(self, image: np.ndarray, include_unclassified: bool = False) -> List[Detection]:
        mask = dark_mask(image, self.cfg)
        rings = find_rings(mask, self.cfg)
        out: List[Detection] = []
        for ring in rings:
            feats = extract_features(mask, ring, self.cfg)
            if feats is None:
                if include_unclassified:
                    out.append(Detection(None, None, ring, None))
                continue
            pts = [(float(x), float(y)) for x, y in feats.mouth.points]
            an = mouth_analysis(feats.eye1, feats.eye2, pts)
            if an is None:
                if include_unclassified:
                    out.append(Detection(None, None, ring, feats))
                continue
            s = an["sagitta"]
            out.append(Detection(label_for(s, self.cfg.sagitta_threshold), s, ring, feats, an))
        # Orden de izquierda a derecha, arriba a abajo.
        out.sort(key=lambda d: (round(d.ring.cy / max(1.0, d.ring.r * 2)), d.ring.cx))
        return out
