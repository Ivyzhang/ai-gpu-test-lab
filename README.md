# Multi-Workload TensorRT Inference Quality Gate

This repository runs the same GPU inference quality gate against two workloads:

```text
tiny-transformer                 local deterministic baseline, hidden=256
distilbert-base-uncased          open-source Transformer Encoder, hidden=768
```

Each workload is tested independently through:

```text
PyTorch reference -> ONNX -> ONNX Runtime CPU -> TensorRT FP16 Engine -> GPU gates
```

Outputs from the two different models are never compared directly. Cross-workload
comparison is limited to latency, throughput, memory and Engine size.

## Contract

Both workloads use:

```text
input_ids       int32 [batch, sequence]
attention_mask  int32 [batch, sequence], values 0 or 1
```

The TensorRT profile is `min=(1,1)`, `opt=(4,128)`, `max=(8,512)`.

| Workload | Output |
| --- | --- |
| `tiny-transformer` | `hidden_states`, float16 `[B,S,256]` |
| `distilbert-base-uncased` | `hidden_states`, float16 `[B,S,768]` |

`sequence=1` is a short Encoder case, not an autoregressive decoder benchmark.

## Run

```bash
docker build -t inference-quality-gate .
```

CPU and ONNX Runtime checks:

```bash
docker run --rm inference-quality-gate \
  python scripts/run_suite.py --suite operator,onnx
```

Full GPU gate:

```bash
docker run --rm --gpus all -v "$PWD:/workspace" \
  inference-quality-gate \
  python scripts/run_suite.py \
  --suite operator,triton,onnx,correctness,dynamic,negative,performance,soak
```

Pin the DistilBERT revision in CI:

```bash
export DISTILBERT_REVISION=<fixed-huggingface-revision>
```

Approve a complete workload baseline only after review:

```bash
python scripts/run_suite.py --suite performance \
  --approve-baseline baselines/two-workloads-a100.json
```

Compare later runs:

```bash
python scripts/run_suite.py --suite performance \
  --baseline baselines/two-workloads-a100.json
```

## Test layers

| Suite | Coverage |
| --- | --- |
| `operator` | Shared MatMul, LayerNorm, Softmax and GELU checks |
| `triton` | Independent Triton Softmax correctness and timing |
| `onnx` | PyTorch versus ONNX Runtime for both workloads and dynamic shapes |
| `correctness` | PyTorch, ONNX Runtime and TensorRT output comparison per workload |
| `dynamic` | Profile boundary and non-aligned shapes per workload |
| `negative` | Invalid dtype, token, mask, shape, tensor name and Engine |
| `performance` | Five CUDA Event trials per workload case |
| `soak` | 1000 iterations, output drift and resource growth |

Workload cases are:

```text
short-b1-s1, short-b4-s1, short-b8-s1
seq-b1-s8, seq-b4-s128, seq-b8-s512
```

Performance report keys include the workload name:

```text
tiny-transformer::seq-b4-s128
distilbert-base-uncased::seq-b4-s128
```

## Reports

Each run writes:

```text
reports/<run-id>/summary.json
reports/<run-id>/junit.xml
reports/<run-id>/release-summary.md
reports/<run-id>/artifacts/<workload>.onnx
reports/<run-id>/artifacts/<workload>.plan
```

Reports record workload identity, model revision, artifact hashes, environment fingerprint,
correctness metrics and raw performance samples. Performance comparison is valid only for
the same workload set, model revisions, GPU, driver, CUDA, TensorRT, PyTorch and image.
Different environments or case sets are `NOT_COMPARABLE`. A median regression above 8%
with non-overlapping bootstrap intervals is `PERFORMANCE_REVIEW_REQUIRED`.

## Requirements

- `requirements.txt`: PyTorch, Transformers, ONNX, ONNX Runtime and pytest.
- `requirements-gpu-linux.txt`: Triton and TensorRT for Linux x86_64.
- GPU execution requires NVIDIA Container Toolkit and a TensorRT-supported GPU.

The DistilBERT model and tokenizer are loaded from the Hugging Face cache. Production and
CI runs should pin and pre-cache the selected revision.
