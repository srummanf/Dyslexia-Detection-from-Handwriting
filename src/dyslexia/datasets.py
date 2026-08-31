"""Dataset loading helpers."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from dyslexia.config import load_config
from dyslexia.features import FEATURE_NAMES

_CSV_COLUMN_ALIASES = {
    "spelling_accuracy": "spelling_accuracy",
    "gramatical_accuracy": "grammatical_accuracy",
    "grammatical_accuracy": "grammatical_accuracy",
    "percentage_of_corrections": "percentage_of_corrections",
    "percentage_of_phonetic_accuraccy": "phonetic_accuracy",
    "phonetic_accuracy": "phonetic_accuracy",
    "presence_of_dyslexia": "presence_of_dyslexia",
}


def load_linguistic_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the linguistic-feature CSV, normalising the historical (misspelled,
    space-padded) column names to :data:`FEATURE_NAMES` + ``presence_of_dyslexia``.
    """
    cfg = load_config()
    path = path or cfg["paths"]["linguistic_features_csv"]
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: _CSV_COLUMN_ALIASES.get(c, c) for c in df.columns})
    keep = list(FEATURE_NAMES) + ["presence_of_dyslexia"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df[keep].astype(float).assign(
        presence_of_dyslexia=lambda d: d["presence_of_dyslexia"].astype(int)
    )


def iter_sample_images(split: str = "all") -> list[tuple[Path, int]]:
    """Return ``(path, label)`` pairs from ``data/samples/<split>`` where
    label is 1 for ``dyslexic`` and 0 for ``non_dyslexic``."""
    cfg = load_config()
    root = Path(cfg["paths"]["samples_dir"]) / split
    pairs: list[tuple[Path, int]] = []
    for label_name, label in (("dyslexic", 1), ("non_dyslexic", 0)):
        folder = root / label_name
        if folder.is_dir():
            pairs += [(p, label) for p in sorted(folder.glob("*.jpg"))]
    return pairs


def prepare_gambo(zip_path: str | Path | None = None, dest: str | Path | None = None) -> Path:
    """Extract ``Gambo.zip`` into ``data/gambo/`` (idempotent). Returns the
    directory containing ``Train/`` and ``Test/``."""
    cfg = load_config()
    zip_path = Path(zip_path or cfg["paths"]["gambo_zip"])
    dest = Path(dest or cfg["paths"]["gambo_dir"])
    if (dest / "Gambo").exists() or (dest / "Train").exists():
        return dest
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Download the Kaggle 'Dyslexia Handwriting "
            f"Dataset' and place it there."
        )
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest
