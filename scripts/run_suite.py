"""Run selected suites and produce an auditable release decision."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.contract import PROFILE_MAX, PROFILE_MIN, PROFILE_OPT

SUITE_TO_TEST_FILES = {
    "operator": [
        "tests/test_operator_matrix.py",
        "tests/test_operator_gradients.py",
        "tests/test_operator_boundary.py",
    ],
    "triton": ["tests/test_triton_kernel.py"],
    "correctness": ["tests/test_correctness.py"],
    "dynamic": ["tests/test_dynamic_shapes.py"],
    "negative": ["tests/test_negative_cases.py"],
    "performance": ["tests/test_performance.py"],
    "soak": ["tests/test_soak.py"],
}


def _command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
    except Exception:
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_environment() -> dict:
    try:
        import torch

        cuda_runtime = torch.version.cuda or "unavailable"
    except Exception:
        cuda_runtime = "unavailable"
    environment = {
        "gpu_name": _command_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        "driver": _command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "cuda_runtime": cuda_runtime,
        "tensorrt": _module_version("tensorrt"),
        "torch": _module_version("torch"),
        "container_image": os.environ.get("CONTAINER_IMAGE_DIGEST", "unavailable"),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
    }
    comparable = {key: environment[key] for key in (
        "gpu_name", "driver", "cuda_runtime", "tensorrt", "torch", "container_image"
    )}
    environment["fingerprint"] = hashlib.sha256(
        json.dumps(comparable, sort_keys=True).encode()
    ).hexdigest()[:16]
    return environment


def validate_suites(suite_names: list[str]) -> None:
    unknown = sorted(set(suite_names) - set(SUITE_TO_TEST_FILES))
    if unknown:
        raise ValueError(f"unknown suites: {', '.join(unknown)}")


def run_pytest(suite_names: list[str], report_dir: Path) -> int:
    test_files = [file for suite in suite_names for file in SUITE_TO_TEST_FILES[suite]]
    command = [
        sys.executable,
        "-m",
        "pytest",
        *test_files,
        f"--junitxml={report_dir / 'junit.xml'}",
        "-v",
    ]
    environment = os.environ.copy()
    environment["REPORT_DIR"] = str(report_dir.resolve())
    return subprocess.run(command, check=False, env=environment).returncode


def load_candidate_performance(report_dir: Path) -> dict:
    result = {}
    for path in sorted((report_dir / "performance").glob("candidate-*.json")):
        shape = path.stem.removeprefix("candidate-")
        result[shape] = json.loads(path.read_text())
    return result


def evaluate_performance(
    candidate: dict,
    baseline_path: Path | None,
    environment: dict,
) -> tuple[str, str, dict]:
    if not candidate:
        return "PASS", "no performance samples were requested", {}
    if baseline_path is None or not baseline_path.exists():
        return "PASS", "candidate performance recorded; no approved baseline supplied", {
            "candidate": candidate,
        }

    baseline = json.loads(baseline_path.read_text())
    if baseline.get("environment", {}).get("fingerprint") != environment["fingerprint"]:
        return "NOT_COMPARABLE", "baseline environment fingerprint differs", {
            "candidate": candidate,
            "baseline_path": str(baseline_path),
        }

    regressions = []
    missing_cases = []
    for shape, current in candidate.items():
        approved = baseline.get("cases", {}).get(shape)
        if approved is None:
            missing_cases.append(shape)
            continue
        ratio = current["median"] / approved["median"]
        if ratio > 1.08:
            regressions.append({
                "shape": shape,
                "candidate_median_ms": current["median"],
                "baseline_median_ms": approved["median"],
                "ratio": ratio,
            })
    if missing_cases:
        return "NOT_COMPARABLE", "baseline is missing approved cases", {
            "missing_cases": sorted(missing_cases),
            "candidate": candidate,
            "baseline_path": str(baseline_path),
        }
    if regressions:
        return "BLOCKED", "one or more median latencies regressed by more than 8%", {
            "regressions": regressions,
            "baseline_path": str(baseline_path),
        }
    return "PASS", "candidate performance is within the approved baseline", {
        "baseline_path": str(baseline_path),
    }


def generate_release_summary(summary: dict, report_dir: Path, reproduce: str) -> None:
    lines = [
        f"Release decision: {summary['status']}",
        f"Reason: {summary['reason']}",
        f"Environment fingerprint: {summary['environment']['fingerprint']}",
        f"Reproduce: {reproduce}",
        f"Raw report: {report_dir / 'summary.json'}",
    ]
    regressions = summary.get("performance", {}).get("regressions", [])
    if regressions:
        lines.append("Performance regressions:")
        lines.extend(
            f"- {item['shape']}: {item['candidate_median_ms']:.4f} ms vs "
            f"{item['baseline_median_ms']:.4f} ms ({item['ratio']:.3f}x)"
            for item in regressions
        )
    (report_dir / "release-summary.md").write_text("\n".join(lines) + "\n")


def approve_baseline(
    path: Path,
    environment: dict,
    candidate: dict,
) -> None:
    if not candidate:
        raise ValueError(
            "cannot approve a baseline without performance samples"
        )

    compact_cases = {}

    for shape, data in candidate.items():
        compact_cases[shape] = {
            "shape": data.get("shape"),
            "median": data["median"],
            "p95": data["p95"],
            "mean": data["mean"],
            "stdev": data["stdev"],
        }

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "cases": compact_cases,
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="operator", help="comma-separated: " + ",".join(SUITE_TO_TEST_FILES))
    parser.add_argument("--baseline", type=Path, help="approved baseline JSON")
    parser.add_argument("--approve-baseline", type=Path, help="write this successful run as an approved baseline")
    parser.add_argument("--model", type=Path, default=Path("model.onnx"))
    parser.add_argument("--engine", type=Path, default=Path("model.plan"))
    args = parser.parse_args()

    suites = [item.strip() for item in args.suite.split(",") if item.strip()]
    validate_suites(suites)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = Path("reports") / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    pytest_exit_code = run_pytest(suites, report_dir)
    environment = collect_environment()
    candidate = load_candidate_performance(report_dir)
    performance_status, performance_reason, performance_details = evaluate_performance(
        candidate, args.baseline, environment
    )

    if pytest_exit_code != 0:
        status = "BLOCKED"
        reason = "one or more suites failed; see junit.xml"
    else:
        status = performance_status
        reason = performance_reason if candidate else "all requested suites passed"

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
            "model_sha256": sha256_file(args.model) or sha256_file(report_dir / "artifacts/model.onnx"),
            "engine_sha256": sha256_file(args.engine) or sha256_file(report_dir / "artifacts/model.plan"),
        },
        "contract": {
            "profile_min": PROFILE_MIN,
            "profile_opt": PROFILE_OPT,
            "profile_max": PROFILE_MAX,
        },
        "performance": performance_details,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    reproduce = f"python scripts/run_suite.py --suite {args.suite}"
    if args.baseline:
        reproduce += f" --baseline {args.baseline}"
    generate_release_summary(summary, report_dir, reproduce)
    print(f"[{status}] report written to {report_dir}")
    return 1 if status == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
