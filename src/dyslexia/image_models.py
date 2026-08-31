"""Image-based classifiers.

`YoloSampleClassifier`  - whole handwriting sample -> P(dyslexic).  Reuses the
                          YOLOv8n-cls weights trained in the 2024 project
                          (models/image/yolo_samples_best*.pt).
`GamboLetterClassifier` - single hand-written letter -> {Normal, Reversal,
                          Corrected}.  Trained in the notebook on the Gambo
                          "Dyslexia Handwriting Dataset" with a timm backbone.

Both degrade to ``available = False`` (rather than raising on import) when their
optional dependencies or weights are missing, so the app can still start.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from dyslexia.ocr import ImageInput, _to_numpy


class YoloSampleClassifier:
    positive_class = "dyslexic"

    def __init__(self, weights: str | Path):
        self.weights = Path(weights)
        self.available = False
        self._model = None
        self._error: str | None = None
        try:
            from ultralytics import YOLO

            if not self.weights.exists():
                raise FileNotFoundError(self.weights)
            self._model = YOLO(str(self.weights))
            self.available = True
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"

    def predict_proba(self, image: ImageInput) -> float:
        """P(dyslexic) in [0, 1]."""
        if not self.available:
            raise RuntimeError(f"YOLO classifier unavailable ({self._error})")
        result = self._model.predict(_to_numpy(image), verbose=False)[0]
        names = result.names
        probs = result.probs.data.tolist()
        by_name = {names[i]: p for i, p in enumerate(probs)}
        return float(by_name.get(self.positive_class, 1.0 - max(probs)))

    def predict(self, image: ImageInput) -> dict[str, float]:
        if not self.available:
            raise RuntimeError(f"YOLO classifier unavailable ({self._error})")
        result = self._model.predict(_to_numpy(image), verbose=False)[0]
        return {result.names[i]: float(p) for i, p in enumerate(result.probs.data.tolist())}


class GamboLetterClassifier:
    def __init__(
        self,
        weights: str | Path,
        classes: list[str] | None = None,
        backbone: str = "convnext_tiny",
        img_size: int = 224,
    ):
        self.weights = Path(weights)
        self.classes = classes or ["Corrected", "Normal", "Reversal"]
        self.img_size = img_size
        self.available = False
        self._model = None
        self._error: str | None = None
        try:
            import timm
            import torch

            if not self.weights.exists():
                raise FileNotFoundError(self.weights)
            self._torch = torch
            model = timm.create_model(
                backbone, pretrained=False, num_classes=len(self.classes)
            )
            state = torch.load(self.weights, map_location="cpu")
            model.load_state_dict(state.get("model", state))
            model.eval()
            self._model = model
            self.available = True
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"

    def _preprocess(self, image: ImageInput):
        arr = _to_numpy(image)
        pil = Image.fromarray(arr).convert("RGB").resize((self.img_size, self.img_size))
        x = np.asarray(pil, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        return self._torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)

    def predict(self, image: ImageInput) -> dict[str, float]:
        if not self.available:
            raise RuntimeError(f"Gambo classifier unavailable ({self._error})")
        with self._torch.no_grad():
            logits = self._model(self._preprocess(image))
            probs = self._torch.softmax(logits, dim=1)[0].tolist()
        return dict(zip(self.classes, (float(p) for p in probs)))

    def error_probability(self, image: ImageInput) -> float:
        """P(letter shows a dyslexia-typical error) = 1 - P(Normal)."""
        return 1.0 - self.predict(image).get("Normal", 1.0)
