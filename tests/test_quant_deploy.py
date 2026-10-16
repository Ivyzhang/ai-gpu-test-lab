"""Deployment INT8 gate: ONNX QDQ versus the FP32 block. CPU only, no TensorRT."""
from pathlib import Path

import pytest
import torch

from src.quant_model import TinyQDQBlock
from src.quant_onnx import calibration_scales, export_qdq, run_qdq

REL_L2_BUDGET = 0.08
HIDDEN = 32


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = expected.float().norm().clamp_min(1e-12)
    return float((actual.float() - expected.float()).norm() / denom)


@pytest.fixture()
def calibrated(tmp_path: Path):
    g = torch.Generator().manual_seed(11)
    model = TinyQDQBlock(hidden=HIDDEN, seed=3).eval()
    calibration = torch.randn(8, HIDDEN, generator=g)
    path = tmp_path / "tiny.qdq.onnx"
    meta = export_qdq(path, model, calibration)
    return model, calibration, path, meta


def test_qdq_graph_records_scales_and_matches_fp32_on_calibration(calibrated):
    model, calibration, path, meta = calibrated
    assert meta["x_scale"] > 0 and meta["w_scale"] > 0 and meta["y_scale"] > 0
    assert set(calibration_scales(model, calibration)) == {"x_scale", "w_scale", "y_scale"}

    actual = run_qdq(path, calibration)
    expected = model.forward_fp32(calibration)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert _relative_l2(actual, expected) < REL_L2_BUDGET
    # ReLU contract survives quantization: no negative outputs.
    assert float(actual.min()) >= 0.0


def test_same_calibration_exports_the_same_scales(tmp_path: Path):
    model = TinyQDQBlock(hidden=HIDDEN, seed=3).eval()
    calibration = torch.randn(8, HIDDEN, generator=torch.Generator().manual_seed(11))
    first = export_qdq(tmp_path / "a.onnx", model, calibration)
    second = export_qdq(tmp_path / "b.onnx", model, calibration)
    assert first["x_scale"] == second["x_scale"]
    assert first["w_scale"] == second["w_scale"]
    assert first["y_scale"] == second["y_scale"]


def test_input_far_outside_calibration_is_finite_but_not_close(calibrated):
    model, _, path, _ = calibrated
    exploded = torch.full((8, HIDDEN), 50.0)
    actual = run_qdq(path, exploded)
    expected = model.forward_fp32(exploded)
    assert torch.isfinite(actual).all()
    # Scales were fixed on a normal sample. A 50x peak must saturate, so the
    # FP32 reference is not a meaningful allclose target. The gate records that
    # instead of widening the tolerance until it passes.
    assert _relative_l2(actual, expected) > REL_L2_BUDGET


def test_zero_calibration_is_rejected(tmp_path: Path):
    model = TinyQDQBlock(hidden=HIDDEN, seed=1).eval()
    with pytest.raises(ValueError, match="non-zero"):
        export_qdq(tmp_path / "zero.onnx", model, torch.zeros(4, HIDDEN))
