import json

from scripts.run_suite import evaluate_performance


ENVIRONMENT = {"fingerprint": "same-environment"}
CANDIDATE = {"4x128": {"median": 1.0}}


def _write_baseline(tmp_path, fingerprint="same-environment", median=1.0):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({
        "environment": {"fingerprint": fingerprint},
        "cases": {"4x128": {"median": median}},
    }))
    return path


def test_candidate_without_baseline_is_recorded_not_blocked():
    status, reason, details = evaluate_performance(CANDIDATE, None, ENVIRONMENT)
    assert status == "PASS"
    assert "no approved baseline" in reason
    assert details["candidate"] == CANDIDATE


def test_environment_mismatch_is_not_comparable(tmp_path):
    baseline = _write_baseline(tmp_path, fingerprint="other-environment")
    status, reason, _ = evaluate_performance(CANDIDATE, baseline, ENVIRONMENT)
    assert status == "NOT_COMPARABLE"
    assert "fingerprint differs" in reason


def test_more_than_eight_percent_regression_blocks(tmp_path):
    baseline = _write_baseline(tmp_path, median=0.90)
    status, reason, details = evaluate_performance(CANDIDATE, baseline, ENVIRONMENT)
    assert status == "BLOCKED"
    assert "more than 8%" in reason
    assert details["regressions"][0]["shape"] == "4x128"


def test_within_threshold_passes(tmp_path):
    baseline = _write_baseline(tmp_path, median=0.95)
    status, reason, _ = evaluate_performance(CANDIDATE, baseline, ENVIRONMENT)
    assert status == "PASS"
    assert "within" in reason
