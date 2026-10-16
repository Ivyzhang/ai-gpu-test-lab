"""Operator-level INT8 gate. CPU only. Reference is FP32 PyTorch, never another INT8 run."""
import torch

from src.quant import QMAX, quantize_linear, quantize_per_tensor, symmetric_scale

REL_L2_BUDGET = 0.05


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = expected.float().norm().clamp_min(1e-12)
    return float((actual.float() - expected.float()).norm() / denom)


def test_zero_tensor_stays_zero():
    q = quantize_per_tensor(torch.zeros(4, 8))
    assert q.scale == 0.0
    assert torch.count_nonzero(q.values) == 0
    assert torch.equal(q.dequant(), torch.zeros(4, 8))


def test_peak_saturates_at_qmax_not_past_it():
    x = torch.tensor([-3.0, 0.0, 1.5, 6.0])
    q = quantize_per_tensor(x)
    assert q.scale == pytest_scale(6.0)
    assert int(q.values.max()) == QMAX
    assert int(q.values.min()) >= -QMAX
    # 6.0 is the peak, so it round-trips. 1.5 is off-grid and does not.
    assert abs(float(q.dequant()[-1]) - 6.0) < 1e-5


def pytest_scale(peak: float) -> float:
    return peak / QMAX


def test_scale_is_deterministic_for_the_same_tensor():
    x = torch.arange(-4, 5, dtype=torch.float32)
    assert symmetric_scale(x) == symmetric_scale(x.clone())
    a = quantize_per_tensor(x)
    b = quantize_per_tensor(x)
    assert a.scale == b.scale
    assert torch.equal(a.values, b.values)


def test_quantized_linear_matches_fp32_within_relative_l2():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(16, 32, generator=g)
    weight = torch.randn(32, 32, generator=g) * 0.05
    bias = torch.randn(32, generator=g) * 0.05
    actual = quantize_linear(x, weight, bias)
    expected = x.float() @ weight.float() + bias.float()
    assert actual.shape == expected.shape
    assert not torch.isnan(actual).any()
    assert _relative_l2(actual, expected) < REL_L2_BUDGET


def test_activation_overflow_is_clipped_instead_of_wrapping():
    weight = torch.eye(4)
    x = torch.full((2, 4), 1000.0)
    actual = quantize_linear(x, weight)
    # Symmetric int8 cannot represent 1000; every output must be finite and capped.
    assert torch.isfinite(actual).all()
    assert float(actual.abs().max()) <= 1000.0
