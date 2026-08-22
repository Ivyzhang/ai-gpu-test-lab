"""Gradient parity between reference(CPU) and candidate(CUDA) for trainable operators."""
import pytest
import torch

from src.operators import OPERATORS

TRAINABLE_OPS = ["matmul", "layernorm", "gelu"]
GRAD_TOLERANCE = dict(atol=1e-2, rtol=1e-2)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]


@pytest.mark.parametrize("op_name", TRAINABLE_OPS)
def test_operator_gradient_matches_reference(op_name):
    op = OPERATORS[op_name](hidden=128)

    # fresh leaf tensors each case: no gradient history to clear before asserting
    x_ref = torch.randn(64, 128, requires_grad=True)
    x_cand = x_ref.detach().clone().to(device="cuda", dtype=torch.float16).requires_grad_()

    op.reference(x_ref).sum().backward()
    op.candidate(x_cand, dtype=torch.float16).sum().backward()

    assert x_ref.grad is not None and x_cand.grad is not None
    assert not torch.isnan(x_cand.grad).any() and not torch.isinf(x_cand.grad).any(), op_name
    torch.testing.assert_close(x_cand.grad.cpu().float(), x_ref.grad.float(), **GRAD_TOLERANCE)
