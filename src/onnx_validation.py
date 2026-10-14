"""CPU-only validation of the exported model's ONNX graph and IO contract."""
from pathlib import Path

import onnx
from onnx import TensorProto

from src.contract import INPUT_NAMES, OUTPUT_NAME


def validate_onnx_io(
    graph_path: str | Path, hidden_size: int, expected_float_type: int
) -> None:
    graph = onnx.load(str(graph_path))
    # Structural checking alone does not reject all mixed input types.
    # Full checking includes shape/type inference before runtime compilation.
    onnx.checker.check_model(graph, full_check=True)
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
    if output.shape.dim[2].dim_value != hidden_size:
        raise RuntimeError(f"{graph_path} output hidden size is not {hidden_size}")
