"""Generate operator test cases for shape x dtype x layout matrix coverage."""
import itertools
from dataclasses import dataclass

import torch

SHAPES = [(128, 128), (1024, 1024), (4096, 4096), (3, 1023)]  # 含非对齐尺寸
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
            x = x.T  # 制造 non-contiguous tensor
        return x


def generate_cases(op_name: str) -> list[OperatorCase]:
    cases = []
    for shape, dtype, layout in itertools.product(SHAPES, DTYPES, LAYOUTS):
        case_id = f"{op_name}-{shape[0]}x{shape[1]}-{dtype}-{layout}"
        cases.append(OperatorCase(case_id, shape, dtype, layout))
    return cases


if __name__ == "__main__":
    print("case generator placeholder")
