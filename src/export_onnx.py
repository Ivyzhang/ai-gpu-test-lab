"""Export workload graphs as separate FP32 and FP16 ONNX artifacts."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import onnx
import torch
from onnx import TensorProto
from onnxconverter_common import float16

from src.contract import INPUT_NAMES, OUTPUT_NAME
from src.workloads import WORKLOADS, WORKLOAD_NAMES


class FP32ExportWrapper(torch.nn.Module):
    """Prevent a model's deployment forward from contaminating the FP32 graph."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model.forward_fp32(input_ids, attention_mask)


def sha256_of_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_fp32_graph(path: Path, spec) -> None:
    model = FP32ExportWrapper(spec.create_model().eval()).eval()
    batch, sequence = spec.profile_opt
    input_ids = torch.randint(0, spec.vocab_size, (batch, sequence), dtype=torch.int32)
    attention_mask = torch.ones(batch, sequence, dtype=torch.int32)

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(path),
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


def _fix_fp16_elementwise_casts(graph: onnx.ModelProto) -> None:
    """Normalize Cast nodes feeding mixed FP16/FP32 elementwise operations.

    ONNX value_info can retain stale types after conversion. TensorRT uses the Cast
    attribute itself, so a Cast(to=FLOAT) feeding a known FP16 operand must be
    changed in the FP16 deployment graph.
    """
    producers = {
        output_name: node
        for node in graph.graph.node
        for output_name in node.output
    }
    tensor_types = {}
    for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output):
        tensor_types[value.name] = value.type.tensor_type.elem_type
    for initializer in graph.graph.initializer:
        tensor_types[initializer.name] = initializer.data_type
    elementwise_ops = {"Add", "Sub", "Mul", "Div", "Where"}

    for node in graph.graph.node:
        if node.op_type not in elementwise_ops:
            continue
        input_producers = [producers.get(input_name) for input_name in node.input]
        input_types = []
        for input_name, producer in zip(node.input, input_producers):
            if producer is not None and producer.op_type == "Cast":
                attribute = next(
                    (item for item in producer.attribute if item.name == "to"),
                    None,
                )
                input_types.append(attribute.i if attribute is not None else None)
            else:
                input_types.append(tensor_types.get(input_name))
        if TensorProto.FLOAT16 not in input_types:
            continue

        cast_nodes = [
            producer
            for producer in input_producers
            if producer is not None and producer.op_type == "Cast"
        ]
        for cast_node in cast_nodes:
            attribute = next(
                (item for item in cast_node.attribute if item.name == "to"),
                None,
            )
            if attribute is not None and attribute.i == TensorProto.FLOAT:
                attribute.i = TensorProto.FLOAT16


def _check_fp16_elementwise_types(graph: onnx.ModelProto) -> None:
    """Fail before TensorRT if an elementwise node still has mixed float inputs."""
    producers = {
        output_name: node
        for node in graph.graph.node
        for output_name in node.output
    }
    tensor_types = {}
    for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output):
        tensor_types[value.name] = value.type.tensor_type.elem_type
    for initializer in graph.graph.initializer:
        tensor_types[initializer.name] = initializer.data_type

    for node in graph.graph.node:
        if node.op_type not in {"Add", "Sub", "Mul", "Div"}:
            continue
        input_types = []
        for input_name in node.input:
            producer = producers.get(input_name)
            if producer is not None and producer.op_type == "Cast":
                attribute = next(
                    (item for item in producer.attribute if item.name == "to"),
                    None,
                )
                input_types.append(attribute.i if attribute is not None else None)
            else:
                input_types.append(tensor_types.get(input_name))
        float_types = {
            value for value in input_types
            if value in {TensorProto.FLOAT, TensorProto.FLOAT16}
        }
        if len(float_types) > 1:
            raise RuntimeError(
                f"FP16 ONNX has mixed float inputs at {node.name}: {input_types}"
            )


def _convert_to_fp16(fp32_path: Path, fp16_path: Path) -> None:
    graph = onnx.load(str(fp32_path))
    graph = float16.convert_float_to_float16(
        graph,
        keep_io_types=False,
        disable_shape_infer=False,
    )
    _fix_fp16_elementwise_casts(graph)
    _check_fp16_elementwise_types(graph)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(fp16_path))


def _check_io(graph_path: Path, spec, expected_float_type: int) -> None:
    graph = onnx.load(str(graph_path))
    onnx.checker.check_model(graph)
    input_names = {value.name for value in graph.graph.input}
    output_names = {value.name for value in graph.graph.output}
    if input_names != set(INPUT_NAMES) or output_names != {OUTPUT_NAME}:
        raise RuntimeError(
            f"unexpected ONNX IO: inputs={input_names}, outputs={output_names}"
        )
    for value in graph.graph.input:
        if value.type.tensor_type.elem_type != TensorProto.INT32:
            raise RuntimeError(f"{graph_path} input {value.name} must be INT32")
    output = graph.graph.output[0].type.tensor_type
    if output.elem_type != expected_float_type:
        raise RuntimeError(
            f"{graph_path} output dtype is {output.elem_type}, "
            f"expected {expected_float_type}"
        )
    if output.shape.dim[2].dim_value != spec.hidden:
        raise RuntimeError(f"{graph_path} output hidden size is not {spec.hidden}")


def export_workload(output_dir: str | Path, workload_name: str) -> dict:
    if workload_name not in WORKLOADS:
        raise ValueError(f"unknown workload: {workload_name}; choose from {WORKLOAD_NAMES}")
    spec = WORKLOADS[workload_name]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fp32_path = output_path / f"{workload_name}.fp32.onnx"
    fp16_path = output_path / f"{workload_name}.fp16.onnx"

    _export_fp32_graph(fp32_path, spec)
    _check_io(fp32_path, spec, TensorProto.FLOAT)
    _convert_to_fp16(fp32_path, fp16_path)
    _check_io(fp16_path, spec, TensorProto.FLOAT16)

    return {
        "workload_name": workload_name,
        "fp32_onnx": fp32_path,
        "fp16_onnx": fp16_path,
        "fp32_sha256": sha256_of_file(fp32_path),
        "fp16_sha256": sha256_of_file(fp16_path),
        "model_name": spec.model_name,
        "model_revision": spec.revision,
        "hidden": spec.hidden,
    }


def export(
    onnx_path: str = "model.onnx",
    workload_name: str = "distilbert-base-uncased",
    precision: str = "fp16",
) -> dict:
    """Compatibility wrapper; use export_workload for both artifacts."""
    destination = Path(onnx_path)
    result = export_workload(destination.parent, workload_name)
    selected = result[f"{precision}_onnx"]
    if selected != destination:
        destination.write_bytes(selected.read_bytes())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--workload", choices=WORKLOAD_NAMES, default="distilbert-base-uncased")
    args = parser.parse_args()
    print(export_workload(args.out_dir, args.workload))

