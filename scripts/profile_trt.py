import argparse
from pathlib import Path

import torch
from src.workloads import WORKLOADS
from src.trt_runner import TrtRunner

parser = argparse.ArgumentParser()
parser.add_argument("--workload", choices=WORKLOADS, default="distilbert-base-uncased")
parser.add_argument("--engine", type=Path)
args = parser.parse_args()

WORKLOAD_NAME = args.workload
spec = WORKLOADS[WORKLOAD_NAME]
engine_path = args.engine or Path("artifacts") / f"{WORKLOAD_NAME}.fp16.plan"
runner = TrtRunner(str(engine_path), WORKLOAD_NAME)

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

