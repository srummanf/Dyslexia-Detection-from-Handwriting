"""Tabular classifier over the four linguistic features.

The 2024 project shipped a hand-transcribed `DecisionTreeClassifier` pasted into
`app.py` as an if/else ladder. This replaces it with a persisted, calibrated
gradient-boosting pipeline and a small model-selection helper.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dyslexia.features import FEATURE_NAMES, LinguisticFeatures


def build_candidates(random_state: int = 42) -> dict[str, Any]:
    """Candidate pipelines evaluated during model selection.

    Each is a full ``Pipeline`` (impute -> scale -> estimator) so they can be
    cross-validated and persisted uniformly.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    def pipe(estimator, scale: bool = True) -> Pipeline:
        steps = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scale", StandardScaler()))
        steps.append(("clf", estimator))
        return Pipeline(steps)

    return {
        "logistic_regression": pipe(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        ),
        "random_forest": pipe(
            RandomForestClassifier(
                n_estimators=300, max_depth=4, class_weight="balanced",
                random_state=random_state,
            ),
            scale=False,
        ),
        "gradient_boosting": pipe(
            GradientBoostingClassifier(random_state=random_state), scale=False
        ),
        "xgboost": pipe(
            XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9,
                colsample_bytree=0.9, eval_metric="logloss", random_state=random_state,
            ),
            scale=False,
        ),
        "svm_rbf": pipe(
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)
        ),
    }


def select_best(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> tuple[str, dict[str, float]]:
    """Cross-validate every candidate; return the winning name and the
    mean score per candidate."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores: dict[str, float] = {}
    for name, model in build_candidates(random_state).items():
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        scores[name] = float(np.mean(cv_scores))
    best = max(scores, key=scores.get)
    return best, scores


@dataclass
class TabularModel:
    """Fitted pipeline plus provenance metadata."""

    pipeline: Any
    feature_names: tuple[str, ...] = FEATURE_NAMES
    algorithm: str = "unknown"
    metrics: dict[str, float] = field(default_factory=dict)
    trained_at: str = ""

    # -- construction -----------------------------------------------------
    @classmethod
    def train(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        algorithm: str | None = None,
        calibrate: bool = True,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> TabularModel:
        from sklearn.calibration import CalibratedClassifierCV

        X = X[list(FEATURE_NAMES)]
        if algorithm is None:
            algorithm, cv_scores = select_best(
                X, y, cv_folds=cv_folds, random_state=random_state
            )
        else:
            cv_scores = {}
        pipeline = build_candidates(random_state)[algorithm]
        if calibrate and len(y) >= cv_folds * 2:
            # sigmoid (Platt) rather than isotonic: far more stable on ~100 rows
            pipeline = CalibratedClassifierCV(pipeline, method="sigmoid", cv=min(cv_folds, 3))
        pipeline.fit(X, y)
        return cls(
            pipeline=pipeline,
            algorithm=algorithm,
            metrics={"cv_" + k: v for k, v in cv_scores.items()},
            trained_at=_dt.datetime.now().isoformat(timespec="seconds"),
        )

    # -- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> TabularModel:
        import joblib

        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a TabularModel")
        return obj

    # -- inference ----------------------------------------------------
    def _frame(self, features: Any) -> pd.DataFrame:
        if isinstance(features, LinguisticFeatures):
            features = features.as_dict()
        if isinstance(features, Mapping):
            return pd.DataFrame([{n: features[n] for n in self.feature_names}])
        if isinstance(features, pd.DataFrame):
            return features[list(self.feature_names)]
        arr = np.atleast_2d(np.asarray(features, dtype=float))
        return pd.DataFrame(arr, columns=list(self.feature_names))

    def predict_proba(self, features: Any) -> np.ndarray:
        """Probability of the positive class (dyslexia present)."""
        return self.pipeline.predict_proba(self._frame(features))[:, 1]

    def predict(self, features: Any, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(int)
