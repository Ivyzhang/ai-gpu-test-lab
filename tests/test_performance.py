"""CUDA-Event device-side latency sampling: produces candidate baselines for release review."""
import json
import math
import os
import statistics
from pathlib import Path

import pytest
import torch

from src.contract import WORKLOAD_CASES

SAMPLE_CASES = WORKLOAD_CASES
WARMUP_ITERS = 100
MEASURED_ITERS = 500
TRIALS = int(os.environ.get("PERFORMANCE_TRIALS", "5"))


def _measure_latency_ms(trt_runner, batch, seq):
    input_ids = torch.randint(0, trt_runner.spec.vocab_size, (batch, seq), device="cuda", dtype=torch.int32)
    attention_mask = torch.ones(batch, seq, device="cuda", dtype=torch.int32)

    for _ in range(WARMUP_ITERS):
        trt_runner.run_async(input_ids, attention_mask)
    trt_runner.stream.synchronize()

    samples = []
    for _ in range(MEASURED_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(trt_runner.stream)
        trt_runner.run_async(input_ids, attention_mask)
        end.record(trt_runner.stream)
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _summarize(samples):
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "median": statistics.median(samples),
        "p95": ordered[p95_index],
        "mean": statistics.mean(samples),
        "stdev": statistics.stdev(samples),
        "raw_samples": samples,
    }


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("case", SAMPLE_CASES, ids=[case.case_id for case in SAMPLE_CASES])
def test_performance_candidate_sample(case, trt_runner, tmp_path, record_property):
    batch, seq = case.shape
    trials = [_summarize(_measure_latency_ms(trt_runner, batch, seq)) for _ in range(TRIALS)]
    trial_medians = [trial["median"] for trial in trials]
    aggregate = _summarize(trial_medians)
    stats = {
        "schema_version": 2,
        "workload_name": trt_runner.spec.name,
        "case_id": case.case_id,
        "mode": case.mode,
        "shape": {"batch": batch, "sequence": seq},
        "median": aggregate["median"],
        "p95": aggregate["p95"],
        "mean": aggregate["mean"],
        "stdev": aggregate["stdev"],
        "trial_medians": trial_medians,
        "trials": trials,
    }
    output_dir = Path(os.environ.get("REPORT_DIR", str(tmp_path))) / "performance"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"candidate-{trt_runner.spec.name}-{case.case_id}.json"
    out_path.write_text(json.dumps(stats))
    record_property("median_ms", stats["median"])
    record_property("p95_ms", stats["p95"])

    # This only checks the measurement itself is valid. The >1.08x regression
    # gate against an *approved* baselines/<gpu>/<image-tag>.json is enforced
    # by scripts/run_suite.py, not here (see test.md "性能测试" §5).
    assert stats["median"] > 0
    assert len(trials) == TRIALS
    assert all(len(trial["raw_samples"]) == MEASURED_ITERS for trial in trials)
