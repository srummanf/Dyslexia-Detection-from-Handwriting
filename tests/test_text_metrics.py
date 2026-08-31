from dyslexia.text_metrics import levenshtein, similarity_ratio


def test_levenshtein_basic():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "") == 3
    assert levenshtein("flaw", "lawn") == 2


def test_levenshtein_symmetric():
    assert levenshtein("dyslexia", "dyslexic") == levenshtein("dyslexic", "dyslexia")


def test_similarity_ratio_bounds():
    assert similarity_ratio("hello world", "hello world") > 0.9
    assert similarity_ratio("hello world", "xxxxx xxxxx") < 0.3
