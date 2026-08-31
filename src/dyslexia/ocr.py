"""Offline OCR backends.

The original project called the Azure Computer Vision Read API with keys
committed to the repo. This module replaces that with pluggable *local*
engines so the app runs with no network access and no secrets.

    engine = get_ocr_engine("easyocr")
    text = engine.read("sample.jpg")
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ImageInput = str | Path | bytes | Image.Image | np.ndarray


def _to_numpy(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, bytes):
        pil = Image.open(io.BytesIO(image))
    else:
        pil = Image.open(image)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return np.asarray(pil)


class OCREngine(ABC):
    """Return recognised text for a handwriting image, joined into one string."""

    name: str = "base"

    @abstractmethod
    def read(self, image: ImageInput) -> str: ...

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<OCREngine {self.name}>"


class NullOCR(OCREngine):
    """No-op engine. Lets the app import and run when no OCR stack is
    installed; feature extraction is then unavailable and the UI says so."""

    name = "null"

    def read(self, image: ImageInput) -> str:
        raise RuntimeError(
            "No OCR engine available. Install 'easyocr' (pip install easyocr) "
            "or set ocr.engine to 'tesseract' with Tesseract on PATH."
        )


class EasyOCREngine(OCREngine):
    """CRAFT + CRNN recogniser from the `easyocr` package (bundled weights,
    fully offline after the first download)."""

    name = "easyocr"

    def __init__(self, languages: list[str] | None = None, min_confidence: float = 0.3):
        import easyocr  # imported lazily; heavy (pulls torch)

        self.min_confidence = min_confidence
        self._reader = easyocr.Reader(languages or ["en"], gpu=False, verbose=False)

    def read(self, image: ImageInput) -> str:
        results = self._reader.readtext(_to_numpy(image), detail=1, paragraph=False)
        words = [text for _box, text, conf in results if conf >= self.min_confidence]
        return " ".join(words).strip()


class TesseractEngine(OCREngine):
    """Wraps `pytesseract`. Requires the Tesseract binary on PATH."""

    name = "tesseract"

    def __init__(self, languages: list[str] | None = None, **_: object):
        import pytesseract

        self._pytesseract = pytesseract
        self._lang = "+".join(languages or ["eng"])

    def read(self, image: ImageInput) -> str:
        pil = Image.fromarray(_to_numpy(image))
        return self._pytesseract.image_to_string(pil, lang=self._lang).strip()


_ENGINES: dict[str, type[OCREngine]] = {
    "easyocr": EasyOCREngine,
    "tesseract": TesseractEngine,
    "null": NullOCR,
}


def get_ocr_engine(
    engine: str = "easyocr",
    *,
    languages: list[str] | None = None,
    min_confidence: float = 0.3,
    fallback_to_null: bool = True,
) -> OCREngine:
    """Instantiate an OCR engine by name, degrading to :class:`NullOCR` if the
    requested backend cannot be imported (unless ``fallback_to_null=False``)."""
    key = (engine or "easyocr").lower()
    cls = _ENGINES.get(key)
    if cls is None:
        raise ValueError(f"Unknown OCR engine {engine!r}. Options: {sorted(_ENGINES)}")
    try:
        if cls is NullOCR:
            return cls()
        return cls(languages=languages, min_confidence=min_confidence)
    except Exception as exc:
        if not fallback_to_null:
            raise
        engine_obj = NullOCR()
        engine_obj._init_error = exc  # type: ignore[attr-defined]
        return engine_obj
