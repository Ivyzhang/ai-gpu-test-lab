"""Generate generic operator cases, separate from model workload cases."""
import itertools
from dataclasses import dataclass

import torch

SHAPES = [(128, 128), (1024, 1024), (4096, 4096), (3, 1023)]  # 含非对齐尺寸
# Explicit GEMM dimensions for tests that need to distinguish M, K and N.
MATMUL_SHAPES = [(128, 128, 128), (1024, 1024, 1024), (3, 1023, 1025)]
DTYPES = [torch.float32, torch.float16]
LAYOUTS = ["contiguous", "transposed"]


@dataclass
class OperatorCase:
    case_id: str
    shape: tuple
    dtype: torch.dtype
    layout: str

    def make_input(self, seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(*self.shape, generator=g)
        if self.layout == "transposed":
            x = torch.randn(self.shape[1], self.shape[0],
                            generator=g).T
            # 制造 non-contiguous tensor
        return x


@dataclass
class MatMulCase:
    case_id: str
    m: int
    k: int
    n: int

    def make_input(self, seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.randn(self.m, self.k, generator=g)


def generate_matmul_cases() -> list[MatMulCase]:
    return [
        MatMulCase(f"matmul-{m}x{k}x{n}", m, k, n)
        for m, k, n in MATMUL_SHAPES
    ]


def generate_cases(op_name: str) -> list[OperatorCase]:
    cases = []
    for shape, dtype, layout in itertools.product(SHAPES, DTYPES, LAYOUTS):
        case_id = f"{op_name}-{shape[0]}x{shape[1]}-{dtype}-{layout}"
        cases.append(OperatorCase(case_id, shape, dtype, layout))
    return cases


if __name__ == "__main__":
    print("case generator placeholder")
