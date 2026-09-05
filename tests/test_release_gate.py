import json

from scripts.run_suite import evaluate_performance


ENVIRONMENT = {"fingerprint": "same-environment"}
CASE = "distilbert-base-uncased::seq-b4-s128"
CANDIDATE = {CASE: {"trial_medians": [1.0, 1.0, 1.0, 1.0, 1.0]}}


def _write_baseline(tmp_path, fingerprint="same-environment", medians=None):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({
        "environment": {"fingerprint": fingerprint},
        "cases": {CASE: {"trial_medians": medians or [1.0] * 5}},
    }))
    return path


def test_candidate_without_baseline_is_recorded_not_blocked():
    status, reason, details = evaluate_performance(CANDIDATE, None, ENVIRONMENT)
    assert status == "PASS"
    assert "no approved baseline" in reason
    assert details["candidate"] == CANDIDATE


def test_environment_mismatch_is_not_comparable(tmp_path):
    status, reason, _ = evaluate_performance(
        CANDIDATE, _write_baseline(tmp_path, fingerprint="other"), ENVIRONMENT
    )
    assert status == "NOT_COMPARABLE"
    assert "fingerprint" in reason


def test_stable_regression_requires_manual_review(tmp_path):
    baseline = _write_baseline(tmp_path, medians=[0.9] * 5)
    status, reason, details = evaluate_performance(CANDIDATE, baseline, ENVIRONMENT)
    assert status == "PERFORMANCE_REVIEW_REQUIRED"
    assert "bootstrap" in reason
    assert details["regressions"][0]["case_id"] == CASE


def test_within_threshold_passes(tmp_path):
    baseline = _write_baseline(tmp_path, medians=[0.95] * 5)
    status, reason, _ = evaluate_performance(CANDIDATE, baseline, ENVIRONMENT)
    assert status == "PASS"
    assert "within" in reason


def test_case_set_mismatch_is_not_comparable(tmp_path):
    baseline = _write_baseline(tmp_path)
    candidate = {"tiny-transformer::short-b1-s1": {"trial_medians": [1.0] * 5}}
    status, reason, details = evaluate_performance(candidate, baseline, ENVIRONMENT)
    assert status == "NOT_COMPARABLE"
    assert "cases differ" in reason
    assert details["missing_cases"] == ["tiny-transformer::short-b1-s1"]
