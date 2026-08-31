"""Linguistic feature extraction from a handwriting image.

Four features, matching the columns the tabular model was trained on:

    spelling_accuracy          how close the writing is to correctly spelled text
    grammatical_accuracy       how few word-level grammatical edits it needs
    percentage_of_corrections  share of tokens a proof-reader would flag
    phonetic_accuracy          how well misspellings still "sound right"

The 2024 project computed these with the Azure Read API, TextBlob, the Bing
Spell-Check API and `abydos`. This is an offline re-implementation:
`easyocr` for text, `pyspellchecker` for corrections, `jellyfish` for phonetics.
The scores land on a comparable 0-100 scale but are *not* identical to the
originals - see notebooks/dyslexia_modeling.ipynb, which can regenerate the
training CSV with this exact pipeline.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np

from dyslexia.ocr import ImageInput, OCREngine, get_ocr_engine
from dyslexia.text_metrics import similarity_ratio

FEATURE_NAMES: tuple[str, ...] = (
    "spelling_accuracy",
    "grammatical_accuracy",
    "percentage_of_corrections",
    "phonetic_accuracy",
)

_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class LinguisticFeatures:
    spelling_accuracy: float
    grammatical_accuracy: float
    percentage_of_corrections: float
    phonetic_accuracy: float
    extracted_text: str = ""

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in FEATURE_NAMES], dtype=float)

    def as_dict(self) -> dict[str, float]:
        return {n: float(getattr(self, n)) for n in FEATURE_NAMES}

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class FeatureExtractor:
    """Turns an image (or raw text) into :class:`LinguisticFeatures`."""

    def __init__(
        self,
        ocr_engine: OCREngine | str = "easyocr",
        *,
        languages: list[str] | None = None,
        min_confidence: float = 0.3,
    ):
        self.ocr = (
            ocr_engine
            if isinstance(ocr_engine, OCREngine)
            else get_ocr_engine(ocr_engine, languages=languages, min_confidence=min_confidence)
        )

    # -- lazily-built helpers -------------------------------------------------
    @cached_property
    def _spell(self):
        from spellchecker import SpellChecker

        return SpellChecker(distance=2)

    @cached_property
    def _phonetic_encoders(self):
        import jellyfish

        return {
            "soundex": (jellyfish.soundex, 0.4),
            "metaphone": (jellyfish.metaphone, 0.4),
            "nysiis": (jellyfish.nysiis, 0.2),
        }

    @cached_property
    def _grammar_tool(self):
        try:
            import language_tool_python

            return language_tool_python.LanguageTool("en-US")
        except Exception:
            return None

    # -- public API ---------------------------------------------------------
    def from_text(self, text: str) -> LinguisticFeatures:
        text = (text or "").strip()
        if not text:
            return LinguisticFeatures(0.0, 0.0, 0.0, 0.0, extracted_text="")
        corrected = self._corrected_text(text)
        return LinguisticFeatures(
            spelling_accuracy=self._spelling_accuracy(text, corrected),
            grammatical_accuracy=self._grammatical_accuracy(corrected),
            percentage_of_corrections=self._percentage_of_corrections(text),
            phonetic_accuracy=self._phonetic_accuracy(text, corrected),
            extracted_text=text,
        )

    def from_image(self, image: ImageInput) -> LinguisticFeatures:
        return self.from_text(self.ocr.read(image))

    __call__ = from_image

    # -- individual features ----------------------------------------------
    def _corrected_text(self, text: str) -> str:
        tokens = _WORD_RE.findall(text)
        if not tokens:
            return text
        unknown = self._spell.unknown(t.lower() for t in tokens)
        fixed = {}
        for tok in tokens:
            low = tok.lower()
            if low in unknown:
                fixed[tok] = self._spell.correction(low) or low
        return _WORD_RE.sub(lambda m: fixed.get(m.group(0), m.group(0)), text)

    def _spelling_accuracy(self, text: str, corrected: str) -> float:
        return max(0.0, similarity_ratio(text, corrected)) * 100.0

    def _grammatical_accuracy(self, corrected_text: str) -> float:
        words = corrected_text.split()
        if not words:
            return 0.0
        if self._grammar_tool is not None:
            matches = [
                m for m in self._grammar_tool.check(corrected_text)
                if m.ruleIssueType != "misspelling"
            ]
            return max(0.0, (len(words) - len(matches)) / (len(words) + 1)) * 100.0
        # Heuristic fallback: penalise repeated words and lowercase sentence starts.
        issues = sum(a.lower() == b.lower() for a, b in zip(words, words[1:]))
        sentences = re.split(r"[.!?]\s+", corrected_text.strip())
        issues += sum(1 for s in sentences if s[:1].islower())
        return max(0.0, (len(words) - issues) / (len(words) + 1)) * 100.0

    def _percentage_of_corrections(self, text: str) -> float:
        tokens = _tokenize(text)
        if not tokens:
            return 0.0
        return len(self._spell.unknown(tokens)) / len(tokens) * 100.0

    def _phonetic_accuracy(self, text: str, corrected: str) -> float:
        score = 0.0
        for encode, weight in self._phonetic_encoders.values():
            a = " ".join(_safe_encode(encode, w) for w in _tokenize(text))
            b = " ".join(_safe_encode(encode, w) for w in _tokenize(corrected))
            score += weight * max(0.0, similarity_ratio(a, b))
        return score * 100.0


def _safe_encode(encode, word: str) -> str:
    try:
        return encode(word) or word
    except Exception:
        return word
