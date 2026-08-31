"""Configuration and path resolution."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# src/dyslexia/config.py -> repo root is two parents up from the package dir.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@functools.lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` and resolve every path under ``paths:`` / ``*_path``
    / ``*_weights`` / ``*_csv`` / ``*_dir`` to an absolute :class:`Path`.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return _resolve_paths(cfg)


def _resolve_paths(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _resolve_path_value(k, v) for k, v in node.items()}
    return node


def _resolve_path_value(key: str, value: Any) -> Any:
    path_like = key.endswith(("_path", "_dir", "_csv", "_weights", "_zip"))
    if path_like and isinstance(value, str):
        candidate = Path(value)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return _resolve_paths(value)


def resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
