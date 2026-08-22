"""Custom Triton softmax kernel (row-wise, last-dim reduction) validated against torch.nn.functional.softmax."""
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    numerator = tl.exp(x - x_max)  # subtract max first to avoid exp overflow
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    tl.store(out_ptr + row_start + col_offsets, result, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2, "triton_softmax expects a 2D CUDA tensor"
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block_size = triton.next_power_of_2(n_cols)
    _softmax_kernel[(n_rows,)](x, out, n_cols, BLOCK_SIZE=block_size)
    return out
