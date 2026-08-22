"""CUDA-Event device-side latency sampling: produces candidate baselines for release review."""
import json
import os
import statistics
from pathlib import Path

import pytest
import torch

from src.contract import PROFILE_MAX, PROFILE_MIN, PROFILE_OPT, VOCAB_SIZE

SAMPLE_SHAPES = [PROFILE_MIN, PROFILE_OPT, PROFILE_MAX]
WARMUP_ITERS = 100
MEASURED_ITERS = 500


def _measure_latency_ms(trt_runner, batch, seq):
    input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq), device="cuda", dtype=torch.int32)
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


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("batch,seq", SAMPLE_SHAPES)
def test_performance_candidate_sample(batch, seq, trt_runner, tmp_path, record_property):
    samples = _measure_latency_ms(trt_runner, batch, seq)
    sorted_samples = sorted(samples)
    stats = {
        "shape": {"batch": batch, "sequence": seq},
        "median": statistics.median(samples),
        "p95": sorted_samples[int(len(sorted_samples) * 0.95)],
        "mean": statistics.mean(samples),
        "stdev": statistics.stdev(samples),
        "raw_samples": samples,
    }
    output_dir = Path(os.environ.get("REPORT_DIR", str(tmp_path))) / "performance"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"candidate-{batch}x{seq}.json"
    out_path.write_text(json.dumps(stats))
    record_property("median_ms", stats["median"])
    record_property("p95_ms", stats["p95"])

    # This only checks the measurement itself is valid. The >1.08x regression
    # gate against an *approved* baselines/<gpu>/<image-tag>.json is enforced
    # by scripts/run_suite.py, not here (see test.md "性能测试" §5).
    assert stats["median"] > 0
    assert len(samples) == MEASURED_ITERS
