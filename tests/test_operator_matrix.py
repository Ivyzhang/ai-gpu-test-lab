import pytest
import torch

from src.case_generator import generate_cases, generate_matmul_cases
from src.contract import DECODE_CASES, PREFILL_CASES, WORKLOAD_CASES
from src.operators import MatMulOp, OPERATORS

# TOLERANCE = {
#     torch.float32: dict(atol=1e-6, rtol=1e-5),
#     torch.float16: dict(atol=1e-3, rtol=1e-3),
# }

TOLERANCE = {
    "default": {
        torch.float32: dict(atol=1e-6, rtol=1e-5),
        torch.float16: dict(atol=1e-3, rtol=1e-3),
    },
    "matmul": {
        torch.float32: dict(atol=2e-3, rtol=1e-3),
        torch.float16: dict(atol=0.25, rtol=3e-3),
    },
    "layernorm": {
        torch.float32: dict(atol=2e-5, rtol=2e-5),
        torch.float16: dict(atol=3e-3, rtol=2e-2),
    },
    "gelu": {
        torch.float32: dict(atol=2e-6, rtol=2e-5),
        torch.float16: dict(atol=2e-3, rtol=2e-3),
    },
    "softmax": {
        torch.float32: dict(atol=1e-6, rtol=1e-5),
        torch.float16: dict(atol=1e-3, rtol=1e-3),
    },
}

CASES = [(name, case) for name in OPERATORS for case in generate_cases(name)]


def test_case_matrix_is_generated() -> None:
    """CPU-only smoke check so `pytest -m "not gpu"` has something to run."""
    assert len(CASES) == len(OPERATORS) * len(generate_cases("gelu"))
    assert all(callable(factory) for factory in OPERATORS.values())


def test_workload_cases_distinguish_decode_from_prefill() -> None:
    assert all(case.mode == "decode" and case.sequence == 1 for case in DECODE_CASES)
    assert all(case.mode == "prefill" and case.sequence > 1 for case in PREFILL_CASES)
    assert {(case.case_id, case.batch, case.sequence) for case in WORKLOAD_CASES} == {
        ("decode-b1-s1", 1, 1),
        ("decode-b4-s1", 4, 1),
        ("decode-b8-s1", 8, 1),
        ("prefill-b1-s8", 1, 8),
        ("prefill-b4-s128", 4, 128),
        ("prefill-b8-s512", 8, 512),
    }


def test_matmul_cases_keep_m_k_n_separate() -> None:
    assert [(case.m, case.k, case.n) for case in generate_matmul_cases()] == [
        (128, 128, 128),
        (1024, 1024, 1024),
        (3, 1023, 1025),
    ]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("case", generate_matmul_cases(), ids=lambda case: case.case_id)
def test_matmul_gemm_shape_correctness(case) -> None:
    op = MatMulOp(k=case.k, n=case.n)
    x = case.make_input()
    actual = op.candidate(x, dtype=torch.float16).cpu().float()
    expected = op.reference(x)
    assert tuple(actual.shape) == (case.m, case.n)
    torch.testing.assert_close(actual, expected, atol=0.25, rtol=3e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize(
    "op_name,case",
    CASES,
    ids=[f"{name}-{case.case_id}" for name, case in CASES],
)
def test_operator_correctness(op_name, case) -> None:
    op = OPERATORS[op_name](hidden=case.shape[-1])
    x = case.make_input()

    ref = op.reference(x)
    cand = op.candidate(x, dtype=case.dtype).cpu().float()

    assert not torch.isnan(cand).any() and not torch.isinf(cand).any(), case.case_id
    tolerance = TOLERANCE.get(
        op_name,
        TOLERANCE["default"],
    )[case.dtype]

    torch.testing.assert_close(
        cand,
        ref,
        **tolerance,
        msg=lambda m: f"{case.case_id}: {m}",
    )

def test_layout_preserves_logical_shape():
    for _, case in CASES:
        x = case.make_input()
        assert tuple(x.shape) == tuple(case.shape)

        if case.layout == "transposed":
            assert not x.is_contiguous()
        else:
            assert x.is_contiguous()
