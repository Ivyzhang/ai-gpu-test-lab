"""Export the tiny block as an INT8 QDQ graph and run it with ONNX Runtime.

Scales use the same symmetric per-tensor rule as src/quant.py. They are baked
from a calibration tensor, so a later input with a larger peak is expected to
saturate. That difference is part of the contract, not a flake.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper

from src.quant import QMAX, symmetric_scale
from src.quant_model import TinyQDQBlock


def _scale_initializer(name: str, scale: float) -> onnx.TensorProto:
    value = np.array([scale], dtype=np.float32)
    return numpy_helper.from_array(value, name=name)


def calibration_scales(model: TinyQDQBlock, example: torch.Tensor) -> dict[str, float]:
    weight = model.linear.weight.detach().float()
    with torch.no_grad():
        pre = example.float() @ weight.t() + model.linear.bias.float()
    scales = {
        "x_scale": symmetric_scale(example),
        "w_scale": symmetric_scale(weight),
        "y_scale": symmetric_scale(pre),
    }
    if min(scales.values()) <= 0:
        raise ValueError("calibration tensor and weight must be non-zero")
    return scales


def export_qdq(path: Path, model: TinyQDQBlock, example: torch.Tensor) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    scales = calibration_scales(model, example)
    weight = model.linear.weight.detach().float().numpy()
    bias = model.linear.bias.detach().float().numpy()
    batch, hidden = example.shape

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, hidden])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, hidden])
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "x_scale"], ["x_q"]),
        helper.make_node("DequantizeLinear", ["x_q", "x_scale"], ["x_f"]),
        helper.make_node("QuantizeLinear", ["W", "w_scale"], ["w_q"]),
        helper.make_node("DequantizeLinear", ["w_q", "w_scale"], ["w_f"]),
        helper.make_node("Gemm", ["x_f", "w_f", "B"], ["gemm_y"], transB=1),
        helper.make_node("QuantizeLinear", ["gemm_y", "y_scale"], ["y_q"]),
        helper.make_node("DequantizeLinear", ["y_q", "y_scale"], ["y_f"]),
        helper.make_node("Relu", ["y_f"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_qdq",
        [x],
        [y],
        initializer=[
            _scale_initializer("x_scale", scales["x_scale"]),
            numpy_helper.from_array(weight, "W"),
            _scale_initializer("w_scale", scales["w_scale"]),
            numpy_helper.from_array(bias, "B"),
            _scale_initializer("y_scale", scales["y_scale"]),
        ],
    )
    model_proto = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="quant-int8-gate",
    )
    model_proto.ir_version = 8
    onnx.checker.check_model(model_proto, full_check=True)
    onnx.save(model_proto, str(path))
    return {**scales, "qmax": QMAX, "weight_quant": "per-tensor symmetric"}


def run_qdq(path: Path, x: torch.Tensor) -> torch.Tensor:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.run(None, {"x": x.detach().float().numpy()})[0]
    return torch.from_numpy(output)
