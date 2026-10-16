"""Symmetric per-tensor INT8 quantization used by the operator and deployment gates.

This is a test double of a post-training quantizer, not TensorRT INT8 and not a
calibrated production recipe. The contract is:

    scale = max(abs(x)) / 127
    q = clip(round(x / scale), -127, 127) stored as int8
    dequant(q) = q * scale

Zero is the zero-point. An all-zero tensor has scale 0 and stays zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

QMAX = 127


@dataclass(frozen=True)
class QuantTensor:
    values: torch.Tensor  # int8
    scale: float

    def dequant(self) -> torch.Tensor:
        return self.values.float() * self.scale


def symmetric_scale(x: torch.Tensor) -> float:
    peak = float(x.detach().float().abs().max())
    if peak == 0.0:
        return 0.0
    return peak / QMAX


def quantize_per_tensor(x: torch.Tensor) -> QuantTensor:
    if not x.is_floating_point():
        raise TypeError("quantize_per_tensor expects a floating-point tensor")
    scale = symmetric_scale(x)
    if scale == 0.0:
        return QuantTensor(torch.zeros_like(x, dtype=torch.int8), 0.0)
    rounded = torch.round(x.float() / scale).clamp(-QMAX, QMAX)
    return QuantTensor(rounded.to(torch.int8), scale)


def quantize_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """INT8xINT8 matmul accumulated in int32, then dequantized back to float."""
    qx = quantize_per_tensor(x)
    qw = quantize_per_tensor(weight)
    acc = torch.matmul(qx.values.int(), qw.values.int())
    out = acc.float() * (qx.scale * qw.scale)
    if bias is not None:
        out = out + bias.float()
    return out
