"""Blends the individual signals into one screening result."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from dyslexia.config import load_config
from dyslexia.features import FeatureExtractor, LinguisticFeatures
from dyslexia.image_models import GamboLetterClassifier, YoloSampleClassifier
from dyslexia.ocr import ImageInput
from dyslexia.tabular import TabularModel


@dataclass
class ScreeningResult:
    risk_score: float                       # blended P(dyslexia) in [0, 1]
    label: str                              # "elevated indicators" | "low indicators"
    signals: dict[str, float] = field(default_factory=dict)   # per-model probability
    weights: dict[str, float] = field(default_factory=dict)   # normalised blend weights
    features: LinguisticFeatures | None = None
    extracted_text: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "label": self.label,
            "signals": self.signals,
            "weights": self.weights,
            "features": self.features.as_dict() if self.features else None,
            "extracted_text": self.extracted_text,
            "notes": self.notes,
        }


class DyslexiaScreener:
    """Lazy-loads whichever components are installed and combines their output.

    Nothing here raises if a model is missing - the corresponding signal is
    simply dropped and the remaining weights are renormalised.
    """

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        self._feature_extractor: FeatureExtractor | None = None
        self._tabular: TabularModel | None = None
        self._yolo: YoloSampleClassifier | None = None
        self._gambo: GamboLetterClassifier | None = None
        self._loaded = False

    # -- lazy component access -----------------------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        ocr_cfg = self.cfg.get("ocr", {})
        try:
            self._feature_extractor = FeatureExtractor(
                ocr_cfg.get("engine", "easyocr"),
                languages=ocr_cfg.get("languages"),
                min_confidence=ocr_cfg.get("min_confidence", 0.3),
            )
        except Exception:
            self._feature_extractor = None

        tab_path = self.cfg["tabular"]["model_path"]
        try:
            self._tabular = TabularModel.load(tab_path)
        except Exception:
            self._tabular = None

        img_cfg = self.cfg.get("image", {})
        self._yolo = YoloSampleClassifier(img_cfg.get("yolo_weights", ""))
        self._gambo = GamboLetterClassifier(
            img_cfg.get("gambo_weights", ""),
            classes=img_cfg.get("gambo_classes"),
            img_size=img_cfg.get("img_size", 224),
        )
        self._loaded = True

    @property
    def component_status(self) -> dict[str, bool]:
        self._ensure_loaded()
        return {
            "features": self._feature_extractor is not None
            and getattr(self._feature_extractor.ocr, "name", "null") != "null",
            "tabular": self._tabular is not None,
            "yolo": bool(self._yolo and self._yolo.available),
            "gambo": bool(self._gambo and self._gambo.available),
        }

    # -- main entry point --------------------------------------------
    def screen(
        self,
        image: ImageInput,
        *,
        letter_crops: Sequence[ImageInput] | None = None,
    ) -> ScreeningResult:
        self._ensure_loaded()
        signals: dict[str, float] = {}
        notes: list[str] = []
        features: LinguisticFeatures | None = None
        text = ""

        # 1. linguistic features -> tabular model
        if self._feature_extractor is not None and self._tabular is not None:
            try:
                features = self._feature_extractor.from_image(image)
                text = features.extracted_text
                if text:
                    signals["tabular"] = float(self._tabular.predict_proba(features)[0])
                else:
                    notes.append("OCR found no readable text; linguistic signal skipped.")
            except Exception as exc:
                notes.append(f"Linguistic pipeline error: {exc}")
        else:
            notes.append("Linguistic pipeline unavailable (OCR or tabular model missing).")

        # 2. whole-sample CNN
        if self._yolo and self._yolo.available:
            try:
                signals["yolo"] = self._yolo.predict_proba(image)
            except Exception as exc:
                notes.append(f"YOLO error: {exc}")
        else:
            notes.append("Whole-sample CNN unavailable.")

        # 3. per-letter Gambo model (optional; needs letter crops)
        if self._gambo and self._gambo.available and letter_crops:
            try:
                errs = [self._gambo.error_probability(c) for c in letter_crops]
                signals["gambo"] = sum(errs) / len(errs)
            except Exception as exc:
                notes.append(f"Gambo error: {exc}")
        elif letter_crops and not (self._gambo and self._gambo.available):
            notes.append("Letter-level model unavailable.")

        risk, weights = self._blend(signals)
        threshold = self.cfg["ensemble"].get("decision_threshold", 0.5)
        label = "elevated indicators" if risk >= threshold else "low indicators"
        if not signals:
            label = "inconclusive"
            notes.append("No model produced a usable signal.")

        return ScreeningResult(
            risk_score=risk,
            label=label,
            signals=signals,
            weights=weights,
            features=features,
            extracted_text=text,
            notes=notes,
        )

    def _blend(self, signals: dict[str, float]) -> tuple[float, dict[str, float]]:
        configured = self.cfg["ensemble"]["weights"]
        active = {k: configured.get(k, 0.0) for k in signals}
        total = sum(active.values())
        if total <= 0:
            return (sum(signals.values()) / len(signals) if signals else 0.0), {}
        weights = {k: v / total for k, v in active.items()}
        risk = sum(signals[k] * w for k, w in weights.items())
        return float(risk), weights
