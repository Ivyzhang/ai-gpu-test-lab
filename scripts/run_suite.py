"""Run selected suites and produce an auditable release decision."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.contract import PROFILE_MAX, PROFILE_MIN, PROFILE_OPT
from src.environment import collect_environment, sha256_file

SUITE_TO_TEST_FILES = {
    "operator": [
        "tests/test_operator_matrix.py",
        "tests/test_operator_gradients.py",
        "tests/test_operator_boundary.py",
    ],
    "triton": ["tests/test_triton_kernel.py"],
    "onnx": ["tests/test_onnx_runtime.py"],
    "correctness": ["tests/test_correctness.py"],
    "dynamic": ["tests/test_dynamic_shapes.py"],
    "negative": ["tests/test_negative_cases.py"],
    "performance": ["tests/test_performance.py"],
    "soak": ["tests/test_soak.py"],
}


def validate_suites(suite_names: list[str]) -> None:
    unknown = sorted(set(suite_names) - set(SUITE_TO_TEST_FILES))
    if unknown:
        raise ValueError(f"unknown suites: {', '.join(unknown)}")


def run_pytest(suite_names: list[str], report_dir: Path) -> int:
    files = [file for suite in suite_names for file in SUITE_TO_TEST_FILES[suite]]
    command = [
        sys.executable,
        "-m",
        "pytest",
        *files,
        f"--junitxml={report_dir / 'junit.xml'}",
        "-v",
    ]
    environment = os.environ.copy()
    environment["REPORT_DIR"] = str(report_dir.resolve())
    return subprocess.run(command, check=False, env=environment).returncode


def load_candidate_performance(report_dir: Path) -> dict:
    result = {}
    for path in sorted((report_dir / "performance").glob("candidate-*.json")):
        data = json.loads(path.read_text())
        key = f"{data['workload_name']}::{data['case_id']}"
        result[key] = data
    return result


def bootstrap_median_ci(values: list[float], seed: int, samples: int = 2000) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(samples)
    )
    low = medians[max(0, math.ceil(len(medians) * 0.025) - 1)]
    high = medians[min(len(medians) - 1, math.ceil(len(medians) * 0.975) - 1)]
    return low, high


def _trial_medians(data: dict) -> list[float]:
    if data.get("trials"):
        return [float(trial["median"]) for trial in data["trials"]]
    if data.get("trial_medians"):
        return [float(value) for value in data["trial_medians"]]
    return [float(data["median"])]


def evaluate_performance(candidate: dict, baseline_path: Path | None, environment: dict):
    if not candidate:
        return "PASS", "no performance samples were requested", {}
    if baseline_path is None or not baseline_path.exists():
        return "PASS", "candidate performance recorded; no approved baseline supplied", {"candidate": candidate}

    baseline = json.loads(baseline_path.read_text())
    if baseline.get("environment", {}).get("fingerprint") != environment["fingerprint"]:
        return "NOT_COMPARABLE", "baseline environment fingerprint differs", {"baseline_path": str(baseline_path)}

    baseline_cases = set(baseline.get("cases", {}))
    candidate_cases = set(candidate)
    if baseline_cases != candidate_cases:
        return "NOT_COMPARABLE", "baseline and candidate workload cases differ", {
            "missing_cases": sorted(candidate_cases - baseline_cases),
            "candidate_missing_cases": sorted(baseline_cases - candidate_cases),
            "baseline_path": str(baseline_path),
        }

    regressions = []
    for case_id, current in candidate.items():
        approved = baseline["cases"][case_id]
        current_medians = _trial_medians(current)
        approved_medians = _trial_medians(approved)
        current_median = statistics.median(current_medians)
        approved_median = statistics.median(approved_medians)
        ratio = current_median / approved_median
        current_ci = bootstrap_median_ci(current_medians, seed=17)
        approved_ci = bootstrap_median_ci(approved_medians, seed=23)
        if ratio > 1.08 and (current_ci[0] > approved_ci[1] or approved_ci[0] > current_ci[1]):
            regressions.append({
                "case_id": case_id,
                "candidate_median_ms": current_median,
                "baseline_median_ms": approved_median,
                "ratio": ratio,
                "candidate_ci95_ms": current_ci,
                "baseline_ci95_ms": approved_ci,
            })

    if regressions:
        return "PERFORMANCE_REVIEW_REQUIRED", "latency regression exceeded 8% with non-overlapping bootstrap intervals", {
            "regressions": regressions,
            "baseline_path": str(baseline_path),
        }
    return "PASS", "candidate performance is within the approved baseline", {"baseline_path": str(baseline_path)}


def approve_baseline(path: Path, environment: dict, candidate: dict) -> None:
    if not candidate:
        raise ValueError("cannot approve a baseline without performance samples")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 2,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "cases": candidate,
    }, indent=2) + "\n")


def generate_release_summary(summary: dict, report_dir: Path, reproduce: str) -> None:
    lines = [
        f"Release decision: {summary['status']}",
        f"Reason: {summary['reason']}",
        f"Environment fingerprint: {summary['environment']['fingerprint']}",
        f"Reproduce: {reproduce}",
        f"Raw report: {report_dir / 'summary.json'}",
    ]
    for item in summary.get("performance", {}).get("regressions", []):
        lines.append(
            f"- {item['case_id']}: {item['candidate_median_ms']:.4f} ms vs "
            f"{item['baseline_median_ms']:.4f} ms ({item['ratio']:.3f}x)"
        )
    (report_dir / "release-summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="operator")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--approve-baseline", type=Path)
    parser.add_argument("--model", type=Path, default=Path("model.onnx"))
    parser.add_argument("--engine", type=Path, default=Path("model.plan"))
    args = parser.parse_args()

    suites = [item.strip() for item in args.suite.split(",") if item.strip()]
    validate_suites(suites)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = Path("reports") / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    pytest_exit_code = run_pytest(suites, report_dir)
    artifact_dir = report_dir / "artifacts"
    environment = collect_environment()
    candidate = load_candidate_performance(report_dir)
    performance_status, performance_reason, performance_details = evaluate_performance(
        candidate, args.baseline, environment
    )
    status = "BLOCKED" if pytest_exit_code != 0 else performance_status
    reason = "one or more suites failed; see junit.xml" if pytest_exit_code != 0 else performance_reason

    if args.approve_baseline:
        if status != "PASS":
            raise RuntimeError("only a passing run can be approved as a baseline")
        approve_baseline(args.approve_baseline, environment, candidate)

    summary = {
        "run_id": run_id,
        "suites": suites,
        "status": status,
        "reason": reason,
        "pytest_exit_code": pytest_exit_code,
        "environment": environment,
        "artifacts": {
            path.stem: sha256_file(path)
            for path in sorted(artifact_dir.glob("*.onnx"))
        } | {
            path.stem: sha256_file(path)
            for path in sorted(artifact_dir.glob("*.plan"))
        },
        "contract": {"profile_min": PROFILE_MIN, "profile_opt": PROFILE_OPT, "profile_max": PROFILE_MAX},
        "performance": performance_details,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    reproduce = f"python scripts/run_suite.py --suite {args.suite}"
    if args.baseline:
        reproduce += f" --baseline {args.baseline}"
    generate_release_summary(summary, report_dir, reproduce)
    print(f"[{status}] report written to {report_dir}")
    return 1 if status == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
