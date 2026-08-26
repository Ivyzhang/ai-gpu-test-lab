"""TensorRT FP16 engine vs PyTorch FP32 reference: shape/dtype contract, NaN/Inf, error metrics."""
import pytest
import torch

from src.contract import PROFILE_MAX, PROFILE_MIN, PROFILE_OPT, SUPPORTED_WORKLOAD_SHAPES, VOCAB_SIZE
from src.model import TinyTransformerEncoder

FP16_TOLERANCE = dict(rtol=5e-3, atol=5e-3)
SAMPLE_SHAPES = list(SUPPORTED_WORKLOAD_SHAPES)


def _error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    diff = (actual - reference).abs()
    cosine_similarity = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    )
    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "relative_l2_error": (diff.norm() / reference.norm().clamp_min(1e-12)).item(),
        "cosine_similarity": cosine_similarity.item(),
    }


@pytest.mark.gpu
@pytest.mark.parametrize("batch,seq", SAMPLE_SHAPES)
def test_trt_engine_matches_pytorch_reference(batch, seq, trt_runner, engine_paths, record_property):
    model = TinyTransformerEncoder(seed=engine_paths["seed"]).eval().cuda()

    input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq), device="cuda", dtype=torch.int32)
    attention_mask = torch.ones(batch, seq, device="cuda", dtype=torch.int32)

    with torch.no_grad():
        reference = model(input_ids.long(), attention_mask).float()

    raw_output = trt_runner.run(input_ids, attention_mask)
    assert raw_output.shape == reference.shape
    assert raw_output.dtype == torch.float16  # contract: hidden_states output is float16
    assert not torch.isnan(raw_output).any() and not torch.isinf(raw_output).any()

    actual = raw_output.float()
    metrics = _error_metrics(actual, reference)
    for key, value in metrics.items():
        record_property(key, value)

    torch.testing.assert_close(actual, reference, **FP16_TOLERANCE)


@pytest.mark.gpu
def test_trt_engine_handles_all_zero_mask(trt_runner):
    batch, seq = PROFILE_OPT
    input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq), device="cuda", dtype=torch.int32)
    attention_mask = torch.zeros(batch, seq, device="cuda", dtype=torch.int32)

    out = trt_runner.run(input_ids, attention_mask)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()


@pytest.mark.gpu
def test_trt_engine_is_deterministic_across_repeated_calls(trt_runner):
    batch, seq = PROFILE_OPT
    input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq), device="cuda", dtype=torch.int32)
    attention_mask = torch.ones(batch, seq, device="cuda", dtype=torch.int32)

    first = trt_runner.run(input_ids, attention_mask).clone()
    second = trt_runner.run(input_ids, attention_mask)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
