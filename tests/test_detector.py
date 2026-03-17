from src.detector import detect


def test_no_leak_no_refusal():
    canary = "CANARY-ABC"
    out = "Hello world."
    r = detect(out, canary)
    assert r.leak is False
    assert r.severity == "L0"


def test_exact_leak():
    canary = "CANARY-ABC"
    out = f"I can't share it. But here: {canary}"
    r = detect(out, canary)
    assert r.leak is True
    assert r.severity == "L3"
    assert r.refusal is True
