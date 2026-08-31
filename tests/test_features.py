import pytest

from dyslexia.features import FEATURE_NAMES, FeatureExtractor, LinguisticFeatures

pytest.importorskip("spellchecker")
pytest.importorskip("jellyfish")


@pytest.fixture(scope="module")
def extractor():
    # NullOCR is fine: these tests exercise the from_text path only.
    return FeatureExtractor("null")


def test_empty_text_is_all_zero(extractor):
    feats = extractor.from_text("   ")
    assert feats.as_array().tolist() == [0.0, 0.0, 0.0, 0.0]


def test_clean_text_scores_high(extractor):
    feats = extractor.from_text("the quick brown fox jumps over the lazy dog")
    assert feats.spelling_accuracy > 80
    assert feats.percentage_of_corrections < 15


def test_messy_text_scores_lower(extractor):
    clean = extractor.from_text("i went to the shop to buy some bread and milk today")
    messy = extractor.from_text("i whent too the shpo too by som bread adn mikl todya")
    assert messy.spelling_accuracy < clean.spelling_accuracy
    assert messy.percentage_of_corrections > clean.percentage_of_corrections


def test_feature_container_roundtrip():
    feats = LinguisticFeatures(90.0, 99.0, 5.0, 92.0, extracted_text="hi")
    assert list(feats.as_dict()) == list(FEATURE_NAMES)
    assert feats.as_array().shape == (4,)
