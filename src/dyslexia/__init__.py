"""Dyslexia screening from handwriting samples.

Public API:
    load_config            -> parsed config.yaml as a dict
    FeatureExtractor       -> image -> 4 linguistic features
    TabularModel           -> gradient-boosted classifier over those features
    YoloSampleClassifier   -> whole-sample dyslexic / non-dyslexic CNN
    GamboLetterClassifier  -> per-letter Normal / Reversal / Corrected CNN
    DyslexiaScreener       -> blends the signals into a single risk score
"""

from dyslexia.config import PROJECT_ROOT, load_config
from dyslexia.features import FEATURE_NAMES, FeatureExtractor, LinguisticFeatures
from dyslexia.screener import DyslexiaScreener, ScreeningResult
from dyslexia.tabular import TabularModel

__all__ = [
    "FEATURE_NAMES",
    "PROJECT_ROOT",
    "DyslexiaScreener",
    "FeatureExtractor",
    "LinguisticFeatures",
    "ScreeningResult",
    "TabularModel",
    "load_config",
]

__version__ = "2.0.0"
