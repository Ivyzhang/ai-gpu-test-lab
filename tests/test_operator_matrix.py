import pytest
import torch

from src.case_generator import generate_cases
from src.operators import OPERATORS

TOLERANCE = {
    torch.float32: dict(atol=1e-6, rtol=1e-5),
    torch.float16: dict(atol=1e-3, rtol=1e-3),
}

CASES = [(name, case) for name in OPERATORS for case in generate_cases(name)]


def test_case_matrix_is_generated() -> None:
    """CPU-only smoke check so `pytest -m "not gpu"` has something to run."""
    assert len(CASES) == len(OPERATORS) * len(generate_cases("gelu"))
    assert all(callable(factory) for factory in OPERATORS.values())


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
    torch.testing.assert_close(
        cand, ref, **TOLERANCE[case.dtype], msg=lambda m: f"{case.case_id}: {m}"
    )
