"""1000-iteration stability loop with resource-growth detection (not a leak claim, see test.md §6)."""
import pytest
import torch

from src.contract import PROFILE_OPT, VOCAB_SIZE

ITERATIONS = 1000
FREE_MEMORY_GROWTH_TOLERANCE_MB = 50


@pytest.mark.gpu
@pytest.mark.slow
def test_soak_1000_iterations_no_drift_or_growth(trt_runner, record_property):
    batch, seq = PROFILE_OPT
    input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq), device="cuda", dtype=torch.int32)
    attention_mask = torch.ones(batch, seq, device="cuda", dtype=torch.int32)

    torch.cuda.synchronize()
    free_before, _ = torch.cuda.mem_get_info()
    first_output = None

    for i in range(ITERATIONS):
        out = trt_runner.run(input_ids, attention_mask)
        assert not torch.isnan(out).any() and not torch.isinf(out).any(), f"iteration {i}"
        if i == 0:
            first_output = out.clone()
        elif i % 100 == 0:
            torch.testing.assert_close(out, first_output, rtol=1e-3, atol=1e-3)

    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    growth_mb = (free_before - free_after) / (1024**2)
    record_property("free_memory_growth_mb", growth_mb)
    assert growth_mb < FREE_MEMORY_GROWTH_TOLERANCE_MB, f"resource growth detected: {growth_mb:.2f} MB"
