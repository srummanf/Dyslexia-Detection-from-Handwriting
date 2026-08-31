import numpy as np
import pytest

from dyslexia.datasets import load_linguistic_dataset
from dyslexia.features import FEATURE_NAMES, LinguisticFeatures
from dyslexia.tabular import TabularModel

pytest.importorskip("sklearn")
pytest.importorskip("xgboost")


@pytest.fixture(scope="module")
def model():
    df = load_linguistic_dataset()
    return TabularModel.train(
        df[list(FEATURE_NAMES)], df["presence_of_dyslexia"],
        algorithm="random_forest", calibrate=False,
    )


def test_predict_proba_in_unit_interval(model):
    df = load_linguistic_dataset()
    proba = model.predict_proba(df[list(FEATURE_NAMES)])
    assert proba.shape == (len(df),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_accepts_multiple_input_types(model):
    dyslexic_like = {"spelling_accuracy": 90, "grammatical_accuracy": 98,
                     "percentage_of_corrections": 15, "phonetic_accuracy": 92}
    p_dict = model.predict_proba(dyslexic_like)[0]
    p_feats = model.predict_proba(LinguisticFeatures(**dyslexic_like))[0]
    p_arr = model.predict_proba([90, 98, 15, 92])[0]
    assert p_dict == pytest.approx(p_feats) == pytest.approx(p_arr)
    assert p_dict > 0.5  # this profile matches the dyslexic group


def test_save_load_roundtrip(model, tmp_path):
    path = model.save(tmp_path / "m.joblib")
    reloaded = TabularModel.load(path)
    x = [98, 99, 6, 98]
    assert reloaded.predict_proba(x)[0] == pytest.approx(model.predict_proba(x)[0])
