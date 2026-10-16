"""Deployment-shaped INT8 graph: float Linear, then QuantizeLinear/DequantizeLinear.

The float path is the reference. The QDQ path is what a backend would constant-fold
into an INT8 engine. Both stay on CPU so the gate does not depend on TensorRT.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.quant import quantize_per_tensor


class TinyQDQBlock(torch.nn.Module):
    def __init__(self, hidden: int = 32, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.linear = torch.nn.Linear(hidden, hidden)
        with torch.no_grad():
            self.linear.weight.copy_(torch.randn(hidden, hidden, generator=generator) * 0.05)
            self.linear.bias.copy_(torch.randn(hidden, generator=generator) * 0.05)

    def forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(x.float()))

    def forward_qdq(self, x: torch.Tensor) -> torch.Tensor:
        qx = quantize_per_tensor(x)
        qw = quantize_per_tensor(self.linear.weight)
        acc = torch.matmul(qx.values.int(), qw.values.int().t())
        y = acc.float() * (qx.scale * qw.scale) + self.linear.bias.float()
        qy = quantize_per_tensor(y)
        return F.relu(qy.dequant())
