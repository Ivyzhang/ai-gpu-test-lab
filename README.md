# TensorRT Inference Release Quality Gate

A reproducible operator-level → Triton-kernel → TensorRT-engine quality gate for one fixed
workload (`TinyTransformerEncoder`: Embedding, MatMul, LayerNorm, Softmax, GELU). See
[test.md](test.md) for the full design doc and 14-day build plan this repo implements.

## Run it

```bash
docker build -t ai-gpu-test-lab .

# CPU-only smoke (no GPU required)
docker run --rm ai-gpu-test-lab pytest -m "not gpu" -q

# Full GPU suite (requires NVIDIA Container Toolkit + a real GPU)
docker run --rm --gpus all -v "$PWD:/workspace" ai-gpu-test-lab \
  python scripts/run_suite.py --suite operator,triton,correctness,dynamic,negative,performance,soak
```

Approve the first reviewed performance run, then compare later candidates only on the
same recorded environment fingerprint:

```bash
python scripts/run_suite.py --suite performance --approve-baseline baselines/approved.json
python scripts/run_suite.py --suite performance --baseline baselines/approved.json
```

Each run writes `reports/<run-id>/summary.json`, `junit.xml` and `release-summary.md`.
`release-summary.md` is the only file a release owner needs to read to decide `PASS` /
`BLOCKED`; engineers follow its "Reproduce" command and `summary.json` path to dig further.

## Scope (what this does and does not claim)

- Operators covered: `MatMul`, `LayerNorm`, `Softmax`, `GELU` — the four ops the fixed
  `TinyTransformerEncoder` workload actually uses. Not a general PyTorch operator suite.
- One custom Triton kernel (`Softmax`), validated against `torch.nn.functional.softmax`
  before any performance number is trusted.
- One workload (`TinyTransformerEncoder`), one input contract (see `src/contract.py`):
  `input_ids`/`attention_mask` are `int32`, `hidden_states` output is `float16`, with a
  TensorRT optimization profile of `min=(1,8)`, `opt=(4,128)`, `max=(8,512)` (batch, sequence).
- Reference implementation is PyTorch CPU FP32 for the operator matrix, and PyTorch GPU FP32
  `eval()` for the TensorRT layer — not FP64, because real GPU accumulation order and kernel
  fusion already change rounding relative to any "true" mathematical answer.
- Performance comparisons are only meaningful on the same GPU model, driver, CUDA, TensorRT
  and image tag. Anything else is reported as `NOT COMPARABLE`, never silently compared.

## Test layout

| File | Layer | What it checks |
| --- | --- | --- |
| `tests/test_operator_matrix.py` | operator | shape × dtype × layout × device correctness |
| `tests/test_operator_gradients.py` | operator | `.grad` parity for trainable ops (matmul/layernorm/gelu) |
| `tests/test_operator_boundary.py` | operator | zero/extreme inputs, wrong dtype rejection |
| `tests/test_triton_kernel.py` | Triton | correctness, boundary inputs, recorded (not asserted) latency |
| `tests/test_correctness.py` | TensorRT | FP16 engine vs FP32 PyTorch reference, error metrics, determinism |
| `tests/test_dynamic_shapes.py` | TensorRT | in-profile shapes, including non-aligned sizes |
| `tests/test_negative_cases.py` | TensorRT | out-of-profile shapes, bad dtype, corrupted engine file |
| `tests/test_performance.py` | TensorRT | CUDA-Event latency sampling (median/p95/mean/stdev) |
| `tests/test_soak.py` | TensorRT | 1000-iteration stability + device memory growth check |

## Why these numbers, not `==`

FP16 vs FP32 never match exactly, so every comparison uses `torch.testing.assert_close`
with a tolerance that was picked for a reason, not to make a test pass:

- FP32 candidate paths: `atol=1e-6, rtol=1e-5`
- FP16 candidate paths (operator matrix, gradients, TensorRT engine): `atol/rtol` in the
  `1e-3`–`5e-3` range, matching the tolerance documented in [test.md](test.md).

Median/p95 are used instead of a single timing sample because GPU work is asynchronous —
a single Python-side wall-clock read is meaningless; `torch.cuda.Event` timing after a
warmup phase is what actually reflects device-side latency.

## Requirements

- `requirements.txt` — CPU-installable deps (torch, numpy, pytest, onnx).
- `requirements-gpu-linux.txt` — Linux x86_64-only GPU deps (`triton`, `tensorrt-cu12`).
