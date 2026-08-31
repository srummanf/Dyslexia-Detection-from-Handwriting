from dyslexia.datasets import load_linguistic_dataset
from dyslexia.features import FEATURE_NAMES


def test_dataset_columns_normalised():
    df = load_linguistic_dataset()
    assert list(df.columns) == list(FEATURE_NAMES) + ["presence_of_dyslexia"]
    assert set(df["presence_of_dyslexia"].unique()) <= {0, 1}
    assert len(df) > 50


def test_dataset_feature_ranges():
    df = load_linguistic_dataset()
    for col in ("spelling_accuracy", "grammatical_accuracy", "phonetic_accuracy"):
        assert df[col].between(0, 100).all()
