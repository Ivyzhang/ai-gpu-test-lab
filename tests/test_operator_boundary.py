"""Boundary/negative-input behavior for the operator matrix (see test.md §算子测试矩阵)."""
import pytest
import torch

from src.operators import OPERATORS

EXTREME_VALUE = 1e4

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]


@pytest.mark.parametrize("op_name", OPERATORS.keys())
def test_zero_input_produces_no_nan_or_inf(op_name):
    op = OPERATORS[op_name](hidden=128)
    x = torch.zeros(4, 128)
    out = op.candidate(x, dtype=torch.float16)
    assert not torch.isnan(out).any() and not torch.isinf(out).any(), op_name


@pytest.mark.parametrize("op_name", OPERATORS.keys())
def test_extreme_value_input_produces_no_nan_or_inf(op_name):
    op = OPERATORS[op_name](hidden=128)
    x = torch.full((4, 128), EXTREME_VALUE)
    x[:, ::2] = -EXTREME_VALUE  # mix +/- extreme values in the same row
    out = op.candidate(x, dtype=torch.float16)
    assert not torch.isnan(out).any() and not torch.isinf(out).any(), op_name


@pytest.mark.parametrize("op_name", OPERATORS.keys())
def test_wrong_dtype_input_raises(op_name):
    op = OPERATORS[op_name](hidden=128)
    x = torch.randint(0, 10, (4, 128))  # integer tensor: all operators expect floating point
    with pytest.raises((RuntimeError, TypeError)):
        op.candidate(x, dtype=torch.float16)
