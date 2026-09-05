"""Export DistilBERT with dynamic batch/sequence axes and verify its graph."""
import argparse
import hashlib

import onnx
import torch

from src.contract import INPUT_NAMES, OUTPUT_NAME
from src.workloads import WORKLOADS, WORKLOAD_NAMES


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(onnx_path: str = "model.onnx", workload_name: str = "distilbert-base-uncased") -> dict:
    if workload_name not in WORKLOADS:
        raise ValueError(f"unknown workload: {workload_name}; choose from {WORKLOAD_NAMES}")
    spec = WORKLOADS[workload_name]
    model = spec.create_model().eval()
    batch, sequence = spec.profile_opt
    input_ids = torch.randint(0, spec.vocab_size, (batch, sequence), dtype=torch.int32)
    attention_mask = torch.ones(batch, sequence, dtype=torch.int32)

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        input_names=list(INPUT_NAMES),
        output_names=[OUTPUT_NAME],
        dynamic_axes={
            INPUT_NAMES[0]: {0: "batch", 1: "sequence"},
            INPUT_NAMES[1]: {0: "batch", 1: "sequence"},
            OUTPUT_NAME: {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph)
    output_shape = graph.graph.output[0].type.tensor_type.shape
    assert output_shape.dim[2].dim_value == spec.hidden

    return {
        "onnx_sha256": sha256_of_file(onnx_path),
        "workload_name": workload_name,
        "model_name": spec.model_name,
        "model_revision": spec.revision,
        "hidden": spec.hidden,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="model.onnx")
    parser.add_argument("--workload", choices=WORKLOAD_NAMES, default="distilbert-base-uncased")
    args = parser.parse_args()
    print(export(args.out, args.workload))
