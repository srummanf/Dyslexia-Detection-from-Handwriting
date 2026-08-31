from dyslexia.screener import DyslexiaScreener, ScreeningResult


def test_blend_renormalises_missing_signals():
    screener = DyslexiaScreener()
    risk, weights = screener._blend({"tabular": 0.8})
    assert risk == 0.8
    assert weights == {"tabular": 1.0}

    risk, weights = screener._blend({"tabular": 1.0, "yolo": 0.0})
    assert 0.0 <= risk <= 1.0
    assert sum(weights.values()) == 1.0


def test_blend_empty_signals():
    risk, weights = DyslexiaScreener()._blend({})
    assert risk == 0.0 and weights == {}


def test_component_status_keys():
    status = DyslexiaScreener().component_status
    assert set(status) == {"features", "tabular", "yolo", "gambo"}
    assert all(isinstance(v, bool) for v in status.values())


def test_screening_result_serialisable():
    r = ScreeningResult(risk_score=0.5, label="low indicators")
    assert r.as_dict()["risk_score"] == 0.5
