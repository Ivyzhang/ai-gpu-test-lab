"""Correctness, boundary and (non-blocking) performance checks for the Triton softmax kernel."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]

TOLERANCE = {torch.float32: dict(atol=1e-5, rtol=1e-5), torch.float16: dict(atol=1e-3, rtol=1e-3)}
SHAPES = [(1, 8), (4, 128), (8, 4097)]  # 4097 is not a power of 2: exercises the kernel's mask logic


def _triton_softmax():
    from triton_kernels.softmax import triton_softmax

    return triton_softmax


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", SHAPES)
def test_triton_softmax_matches_pytorch(dtype, shape):
    triton_softmax = _triton_softmax()
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    ref = F.softmax(x, dim=-1)
    out = triton_softmax(x)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()
    torch.testing.assert_close(out, ref, **TOLERANCE[dtype])


@pytest.mark.parametrize(
    "make_input",
    [
        lambda: torch.zeros(4, 128, device="cuda", dtype=torch.float16),
        lambda: torch.ones(1, 1, device="cuda", dtype=torch.float16),  # single-element row
        lambda: torch.tensor([[1e4, -1e4, 0.0, 1e4]], device="cuda", dtype=torch.float16),
    ],
    ids=["all_zero", "single_element", "large_value_gap"],
)
def test_triton_softmax_boundary_inputs(make_input):
    triton_softmax = _triton_softmax()
    x = make_input()
    out = triton_softmax(x)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()
    row_sums = out.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-2, rtol=1e-2)


def _benchmark(fn, x, n_iters=200, warmup=50):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


def test_triton_softmax_benchmark_is_recorded(record_property):
    """Records real device-side latency; does not assert which implementation is faster."""
    triton_softmax = _triton_softmax()
    x = torch.randn(32, 4096, device="cuda", dtype=torch.float16)
    triton_ms = _benchmark(triton_softmax, x)
    torch_ms = _benchmark(lambda t: F.softmax(t, dim=-1), x)
    record_property("triton_softmax_ms", triton_ms)
    record_property("torch_softmax_ms", torch_ms)
    assert triton_ms > 0 and torch_ms > 0
