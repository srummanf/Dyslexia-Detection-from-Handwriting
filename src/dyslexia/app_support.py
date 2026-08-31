"""Helpers shared by the Streamlit pages (kept out of the app files so they
stay importable and testable)."""

from __future__ import annotations

import functools

from dyslexia.config import load_config
from dyslexia.datasets import load_linguistic_dataset
from dyslexia.screener import DyslexiaScreener

DISCLAIMER = (
    "This tool is a **research demo**, not a medical device. A dyslexia diagnosis "
    "can only be made by a qualified professional through a full assessment. "
    "Treat every result here as a rough, exploratory indicator."
)

RISK_BANDS = [
    (0.0, 0.34, "Low indicators", "#2e7d32"),
    (0.34, 0.66, "Some indicators", "#f9a825"),
    (0.66, 1.01, "Elevated indicators", "#c62828"),
]


def band_for(score: float) -> tuple[str, str]:
    for lo, hi, name, colour in RISK_BANDS:
        if lo <= score < hi:
            return name, colour
    return "Unknown", "#607d8b"


@functools.lru_cache(maxsize=1)
def get_screener() -> DyslexiaScreener:
    return DyslexiaScreener(load_config())


@functools.lru_cache(maxsize=1)
def get_dataset():
    return load_linguistic_dataset()


def feature_labels() -> dict[str, str]:
    return {
        "spelling_accuracy": "Spelling accuracy",
        "grammatical_accuracy": "Grammatical accuracy",
        "percentage_of_corrections": "Corrections needed (%)",
        "phonetic_accuracy": "Phonetic accuracy",
    }
