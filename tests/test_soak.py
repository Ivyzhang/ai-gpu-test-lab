"""Long-run correctness and resource-growth checks for dynamic Encoder shapes."""
import os

import pytest
import torch

from src.contract import PROFILE_OPT

ITERATIONS = 1000
FREE_MEMORY_GROWTH_TOLERANCE_MB = 50


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            pages = int(statm.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024**2)
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return -1.0


@pytest.mark.gpu
@pytest.mark.slow
def test_soak_no_drift_or_growth(trt_runner, record_property):
    batch, sequence = PROFILE_OPT
    input_ids = torch.randint(0, trt_runner.spec.vocab_size, (batch, sequence), device="cuda", dtype=torch.int32)
    attention_mask = torch.ones(batch, sequence, device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()
    free_before, _ = torch.cuda.mem_get_info()
    rss_before = _rss_mb()
    first_output = None

    for iteration in range(ITERATIONS):
        output = trt_runner.run(input_ids, attention_mask)
        assert output.shape == (batch, sequence, trt_runner.spec.hidden)
        assert torch.isfinite(output).all(), f"iteration {iteration}"
        if first_output is None:
            first_output = output.clone()
        elif iteration % 100 == 0:
            torch.testing.assert_close(output, first_output, atol=1e-3, rtol=1e-3)

    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    record_property("free_memory_growth_mb", (free_before - free_after) / (1024**2))
    record_property("rss_before_mb", rss_before)
    record_property("rss_after_mb", _rss_mb())

    del output, first_output, input_ids, attention_mask
    trt_runner.close()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_final, _ = torch.cuda.mem_get_info()
    final_growth_mb = (free_before - free_final) / (1024**2)
    record_property("free_memory_growth_after_cleanup_mb", final_growth_mb)
    assert final_growth_mb < FREE_MEMORY_GROWTH_TOLERANCE_MB
