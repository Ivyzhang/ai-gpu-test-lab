"""In-profile dynamic shape coverage, including non-aligned batch/sequence sizes."""
import pytest
import torch

from src.contract import (
    PROFILE_MAX,
    PROFILE_MIN,
    PROFILE_OPT,
    SUPPORTED_WORKLOAD_SHAPES,
    VOCAB_SIZE,
)

IN_PROFILE_SHAPES = [*SUPPORTED_WORKLOAD_SHAPES, (3, 127), (5, 200)]


def test_decode_and_prefill_shapes_are_in_profile() -> None:
    assert PROFILE_MIN == (1, 1)
    assert all(
        PROFILE_MIN[0] <= batch <= PROFILE_MAX[0]
        and PROFILE_MIN[1] <= seq <= PROFILE_MAX[1]
        for batch, seq in SUPPORTED_WORKLOAD_SHAPES
    )


@pytest.mark.gpu
@pytest.mark.parametrize("batch,seq", IN_PROFILE_SHAPES)
def test_in_profile_shape_executes_successfully(batch, seq, trt_runner):
    input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq), device="cuda", dtype=torch.int32)
    attention_mask = torch.ones(batch, seq, device="cuda", dtype=torch.int32)

    out = trt_runner.run(input_ids, attention_mask)
    assert out.shape[:2] == (batch, seq)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()
