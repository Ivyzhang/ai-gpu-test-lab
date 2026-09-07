"""Per-workload PyTorch, ONNX Runtime and TensorRT correctness checks."""
import pytest
import torch

from src.contract import SUPPORTED_WORKLOAD_SHAPES
from src.workloads import WORKLOADS, make_inputs, reference_forward

FP16_TOLERANCE = {
    "tiny-transformer": {
        "atol": 5e-3,
        "rtol": 5e-3,
    },
    "distilbert-base-uncased": {
        "atol": 2.5e-2,
        "rtol": 1e-2,
    },
}
SAMPLE_SHAPES = list(SUPPORTED_WORKLOAD_SHAPES)


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    diff = (actual - reference).abs()
    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "relative_l2_error": (diff.norm() / reference.norm().clamp_min(1e-12)).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            actual.flatten(), reference.flatten(), dim=0
        ).item(),
    }


@pytest.mark.gpu
@pytest.mark.parametrize("batch,sequence", SAMPLE_SHAPES)
def test_trt_matches_pytorch_reference(batch, sequence, workload_name, trt_runner, record_property):
    spec = WORKLOADS[workload_name]
    model = spec.create_model().eval().cuda()
    input_ids, attention_mask = make_inputs(spec, batch, sequence, seed=batch * 1000 + sequence, device="cuda")

    with torch.no_grad():
        reference = reference_forward(model, input_ids, attention_mask)

    actual_raw = trt_runner.run(input_ids, attention_mask)
    assert actual_raw.shape == (batch, sequence, spec.hidden)
    assert actual_raw.dtype == torch.float16
    assert torch.isfinite(actual_raw).all()

    for key, value in _metrics(actual_raw.float(), reference).items():
        record_property(f"{workload_name}_{key}", value)
    tolerance = FP16_TOLERANCE[workload_name]

    torch.testing.assert_close(
        actual_raw.float(),
        reference.float(),
        **tolerance,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("batch,sequence", SAMPLE_SHAPES)
def test_trt_matches_onnx_runtime(batch, sequence, workload_name, trt_runner, ort_session):
    spec = WORKLOADS[workload_name]
    input_ids, attention_mask = make_inputs(spec, batch, sequence, seed=batch * 1000 + sequence)
    ort_output = torch.from_numpy(
        ort_session.run(
            ["hidden_states"],
            {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()},
        )[0]
    )
    trt_output = trt_runner.run(input_ids.cuda(), attention_mask.cuda()).cpu()
    tolerance = FP16_TOLERANCE[workload_name]

    torch.testing.assert_close(
        trt_output.float(),
        ort_output.float(),
        **tolerance,
    )


@pytest.mark.gpu
def test_trt_accepts_maximum_token_id(workload_name, trt_runner):
    spec = WORKLOADS[workload_name]
    input_ids = torch.full((1, 1), spec.vocab_size - 1, device="cuda", dtype=torch.int32)
    attention_mask = torch.ones((1, 1), device="cuda", dtype=torch.int32)
    output = trt_runner.run(input_ids, attention_mask)
    assert output.shape == (1, 1, spec.hidden)
    assert torch.isfinite(output).all()


@pytest.mark.gpu
def test_trt_is_deterministic(workload_name, trt_runner):
    spec = WORKLOADS[workload_name]
    input_ids, attention_mask = make_inputs(spec, 4, 128, device="cuda")
    first = trt_runner.run(input_ids, attention_mask).clone()
    second = trt_runner.run(input_ids, attention_mask)
    torch.testing.assert_close(first, second, atol=0, rtol=0)

