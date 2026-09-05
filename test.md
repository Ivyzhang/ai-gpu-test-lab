# Multi-Workload Test Design

## Workloads

The gate runs two independent workloads:

| Workload | Model | Hidden size | Vocabulary |
| --- | --- | ---: | ---: |
| `tiny-transformer` | deterministic local TinyTransformerEncoder | 256 | 32000 |
| `distilbert-base-uncased` | Hugging Face DistilBERT Encoder | 768 | 30522 |

The two models are not compared element by element. Each model is compared across its
own PyTorch, ONNX Runtime and TensorRT paths. Model-to-model comparison is limited to
performance, memory, Engine size and coverage.

## Shared input contract

```text
input_ids       int32 [batch, sequence]
attention_mask  int32 [batch, sequence], values 0 or 1
```

Both models use the TensorRT profile:

```text
min=(1,1), opt=(4,128), max=(8,512)
```

The output shape is workload-specific:

```text
tiny-transformer        [batch, sequence, 256] float16
distilbert-base-uncased [batch, sequence, 768] float16
```

## Conversion chain

```text
workload model
    ↓
PyTorch FP32 reference
    ↓ torch.onnx.export
ONNX graph + onnx.checker
    ↓ ONNX Runtime CPU
ORT numerical reference
    ↓ TensorRT FP16 builder
TensorRT Engine
    ↓ execute_async_v3
GPU output
```

For every workload, the numerical checks are:

```text
PyTorch ≈ ONNX Runtime
PyTorch ≈ TensorRT
ONNX Runtime ≈ TensorRT
```

This separates ONNX export failures from TensorRT parser, precision, binding and stream
failures.

## Shape matrix

Standard cases:

```text
short-b1-s1
short-b4-s1
short-b8-s1
seq-b1-s8
seq-b4-s128
seq-b8-s512
```

Additional dynamic cases are `(3,127)` and `(5,200)`. `sequence=1` means short Encoder
execution; it is not an autoregressive decoder case.

## Correctness checks

For each workload and shape, check output shape and dtype, finite values, maximum and mean
absolute error, relative L2 error, cosine similarity and repeated-call determinism.

The initial FP16 comparison uses `atol=5e-3` and `rtol=5e-3`. Tolerance must be calibrated
per model revision, GPU and TensorRT version. Maximum element-wise relative error is
diagnostic only because reference values near zero make it unstable.

## Boundary and negative cases

Each workload checks token id `0`, token id `vocab_size - 1`, token id `vocab_size`,
negative token id, invalid mask values, wrong dtype, mismatched input shapes, profile-outside
shapes, unknown tensor names and corrupted Engine files.

DistilBERT additionally uses tokenizer-generated text fixtures with padding. The all-zero
mask is not a normal text case; a normal padded input contains valid-token and padding
positions.

## Performance and stability

Each workload case uses 100 warmups and five independent trials with 500 CUDA Event samples
per trial. Reports preserve raw samples, per-trial statistics and trial medians. Baselines
are keyed as `<workload>::<case>` and are valid only on the same GPU/software fingerprint
and model revision.

The performance result is `PASS`, `PERFORMANCE_REVIEW_REQUIRED`, `NOT_COMPARABLE` or
`BLOCKED`. A review is required when the median ratio exceeds 1.08 and bootstrap intervals
do not overlap.

The soak suite runs 1000 iterations at `(4,128)`, checks output drift, and measures free
GPU memory, process RSS and memory after TensorRT runner cleanup.

## Reproducibility

Reports record each workload's model name, revision, ONNX hash, Engine hash, GPU, driver,
CUDA, TensorRT, PyTorch and container information. `DISTILBERT_REVISION` should be pinned
and model/tokenizer assets should be pre-cached in CI.
