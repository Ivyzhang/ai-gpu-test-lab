import torch
from src.workloads import WORKLOADS
from src.trt_runner import TrtRunner

WORKLOAD_NAME = "distilbert-base-uncased"
spec = WORKLOADS[WORKLOAD_NAME]
runner = TrtRunner("model.plan", WORKLOAD_NAME)

batch, seq = 4, 128

input_ids = torch.randint(
    0,
    spec.vocab_size,
    (batch, seq),
    device="cuda",
    dtype=torch.int32,
)

attention_mask = torch.ones(
    batch,
    seq,
    device="cuda",
    dtype=torch.int32,
)

# Warmup
for _ in range(20):
    runner.run(input_ids, attention_mask)

torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as profiler:
    for _ in range(20):
        runner.run(input_ids, attention_mask)

print(
    profiler.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30,
    )
)

profiler.export_chrome_trace(
    "reports/profiler/tensorrt_trace.json"
)
