"""Dynamic batch/sequence coverage for the DistilBERT TensorRT profile."""
import pytest
import torch

from src.contract import PROFILE_MAX, PROFILE_MIN, SUPPORTED_WORKLOAD_SHAPES
from src.workloads import WORKLOADS

IN_PROFILE_SHAPES = [*SUPPORTED_WORKLOAD_SHAPES, (3, 127), (5, 200)]


def test_workload_shapes_are_inside_profile():
    assert PROFILE_MIN == (1, 1)
    assert all(
        PROFILE_MIN[0] <= batch <= PROFILE_MAX[0]
        and PROFILE_MIN[1] <= sequence <= PROFILE_MAX[1]
        for batch, sequence in SUPPORTED_WORKLOAD_SHAPES
    )


@pytest.mark.gpu
@pytest.mark.parametrize("batch,sequence", IN_PROFILE_SHAPES)
def test_in_profile_shape_executes(batch, sequence, trt_runner):
    input_ids = torch.randint(
        0, trt_runner.spec.vocab_size, (batch, sequence), device="cuda", dtype=torch.int32
    )
    attention_mask = torch.ones(batch, sequence, device="cuda", dtype=torch.int32)
    output = trt_runner.run(input_ids, attention_mask)
    assert tuple(output.shape[:2]) == (batch, sequence)
    assert output.shape[2] == WORKLOADS[trt_runner.spec.name].hidden
    assert torch.isfinite(output).all()
