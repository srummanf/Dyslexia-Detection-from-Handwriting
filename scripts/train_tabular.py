"""Train and persist the tabular linguistic-feature model.

    python scripts/train_tabular.py [--algorithm xgboost] [--no-calibrate]
"""

from __future__ import annotations

import argparse

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from dyslexia.config import load_config
from dyslexia.datasets import load_linguistic_dataset
from dyslexia.features import FEATURE_NAMES
from dyslexia.tabular import TabularModel, select_best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", default=None, help="force a specific candidate")
    parser.add_argument("--no-calibrate", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    tcfg = cfg["tabular"]
    df = load_linguistic_dataset()
    X, y = df[list(FEATURE_NAMES)], df[tcfg["target"]]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=tcfg["test_size"], stratify=y, random_state=tcfg["random_state"]
    )

    if args.algorithm is None:
        best, scores = select_best(
            X_tr, y_tr, cv_folds=tcfg["cv_folds"], random_state=tcfg["random_state"]
        )
        print("CV ROC-AUC by candidate:")
        for name, score in sorted(scores.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<22} {score:.3f}")
        print(f"-> selected: {best}")
    else:
        best = args.algorithm

    model = TabularModel.train(
        X_tr, y_tr, algorithm=best, calibrate=not args.no_calibrate,
        cv_folds=tcfg["cv_folds"], random_state=tcfg["random_state"],
    )

    proba = model.predict_proba(X_te)
    print(f"\nHold-out ROC-AUC: {roc_auc_score(y_te, proba):.3f}")
    print(classification_report(y_te, (proba >= 0.5).astype(int), digits=3))

    path = model.save(tcfg["model_path"])
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
