"""Deterministic graph mutations exercise export validation and numerical oracles.

These tests mutate ONNX ModelProto objects, not TensorRT's serialized plan bytes.
Invalid graphs must fail the project's pre-build validator. Valid mutations can
either preserve semantics or change results; both controls are necessary.
"""
import copy
import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from src.onnx_validation import validate_onnx_io


def _reference_graph():
    # y = relu(input_ids[..., None] * 0.25 * attention_mask[..., None] - 1)
    # Uses the project's IO names/types, dynamic B/S and hidden size 1.
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["ids_float"], name="ids_cast", to=TensorProto.FLOAT),
        helper.make_node("Cast", ["attention_mask"], ["mask_float"], name="mask_cast", to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["ids_float", "axes"], ["ids_3d"], name="ids_expand"),
        helper.make_node("Unsqueeze", ["mask_float", "axes"], ["mask_3d"], name="mask_expand"),
        helper.make_node("Mul", ["ids_3d", "scale"], ["scaled"], name="scale_ids"),
        helper.make_node("Mul", ["scaled", "mask_3d"], ["masked"], name="apply_mask"),
        helper.make_node("Add", ["masked", "bias"], ["shifted"], name="add_bias"),
        helper.make_node("Relu", ["shifted"], ["hidden_states"], name="activation"),
    ]
    graph = helper.make_graph(
        nodes, "mutation_control",
        [helper.make_tensor_value_info(name, TensorProto.INT32, ["B", "S"])
         for name in ("input_ids", "attention_mask")],
        [helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, ["B", "S", 1])],
        initializer=[
            numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes"),
            numpy_helper.from_array(np.array(0.25, dtype=np.float32), name="scale"),
            numpy_helper.from_array(np.array(-1.0, dtype=np.float32), name="bias"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)


def _run(path, feeds):
    options = ort.SessionOptions()
    # Isolate graph semantics from ORT optimization choices and thread pools.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return session.run(["hidden_states"], feeds)[0]


@pytest.fixture
def graph_case(tmp_path, request):
    report_root = os.environ.get("REPORT_DIR")
    directory = (
        Path(report_root) / "graph-mutations" / request.node.name
        if report_root else tmp_path
    )
    directory.mkdir(parents=True, exist_ok=True)
    model = _reference_graph()
    feeds = {
        "input_ids": np.array([[0, 2, 4, 6]], dtype=np.int32),
        "attention_mask": np.array([[1, 1, 0, 1]], dtype=np.int32),
    }
    # Hand-derived, independent of the graph builder / runtime under test.
    expected = np.array([[[0.0], [0.0], [0.0], [0.5]]], dtype=np.float32)
    original = directory / "original.onnx"
    onnx.save(model, str(original))
    np.savez(directory / "inputs-and-expected.npz", **feeds, expected=expected)
    (directory / "metadata.json").write_text(json.dumps({
        "case": request.node.name,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "opset": 17,
        "provider": "CPUExecutionProvider",
        "oracle": "relu(input_ids[..., None] * 0.25 * attention_mask[..., None] - 1)",
    }, indent=2) + "\n")
    validate_onnx_io(original, hidden_size=1, expected_float_type=TensorProto.FLOAT)
    actual = _run(original, feeds)
    np.testing.assert_array_equal(actual, expected)
    return model, feeds, expected, directory


def _save_mutation(graph_case, mutate):
    model, feeds, expected, directory = graph_case
    mutant = copy.deepcopy(model)
    mutate(mutant)
    path = directory / "mutated.onnx"
    onnx.save(mutant, str(path))
    return path, feeds, expected, directory


def test_dependent_node_reorder_is_rejected(graph_case):
    def mutate(model):
        nodes = list(model.graph.node)
        # Relu consumes shifted, which Add has not produced yet.
        nodes[-2], nodes[-1] = nodes[-1], nodes[-2]
        del model.graph.node[:]
        model.graph.node.extend(nodes)

    path, _, _, _ = _save_mutation(graph_case, mutate)
    with pytest.raises(onnx.checker.ValidationError, match="topologically sorted"):
        validate_onnx_io(path, 1, TensorProto.FLOAT)


def test_deleted_producer_is_rejected(graph_case):
    def mutate(model):
        del model.graph.node[4]  # apply_mask still consumes scaled.

    path, _, _, _ = _save_mutation(graph_case, mutate)
    with pytest.raises(onnx.checker.ValidationError, match="topologically sorted"):
        validate_onnx_io(path, 1, TensorProto.FLOAT)


def test_duplicate_tensor_producer_is_rejected(graph_case):
    def mutate(model):
        model.graph.node[6].output[0] = "masked"
        model.graph.node[7].input[0] = "masked"

    path, _, _, _ = _save_mutation(graph_case, mutate)
    with pytest.raises(onnx.checker.ValidationError, match="SSA"):
        validate_onnx_io(path, 1, TensorProto.FLOAT)


def test_cast_causing_mixed_elementwise_types_is_rejected(graph_case):
    def mutate(model):
        model.graph.node[5].input[0] = "scaled_half"
        model.graph.node.insert(5, helper.make_node(
            "Cast", ["scaled"], ["scaled_half"], name="injected_half_cast", to=TensorProto.FLOAT16
        ))

    path, _, _, _ = _save_mutation(graph_case, mutate)
    with pytest.raises((onnx.checker.ValidationError, onnx.shape_inference.InferenceError)):
        validate_onnx_io(path, 1, TensorProto.FLOAT)


def test_rewired_operator_order_is_caught_by_numerical_oracle(graph_case):
    def mutate(model):
        # Legal topology, different function: relu(x) - 1 != relu(x - 1).
        del model.graph.node[-2:]
        model.graph.node.extend([
            helper.make_node("Relu", ["masked"], ["positive"], name="activation"),
            helper.make_node("Add", ["positive", "bias"], ["hidden_states"], name="add_bias"),
        ])

    path, feeds, expected, directory = _save_mutation(graph_case, mutate)
    validate_onnx_io(path, 1, TensorProto.FLOAT)
    actual = _run(path, feeds)
    np.save(directory / "mutated-output.npy", actual)
    np.testing.assert_array_equal(actual, [[[-1.0], [-0.5], [-1.0], [0.5]]])
    # The suite passes only if the correctness oracle detects the injected fault.
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_lossy_cast_roundtrip_is_caught_by_numerical_oracle(graph_case):
    def mutate(model):
        model.graph.node[6].input[0] = "masked_restored"
        model.graph.node.insert(6, helper.make_node(
            "Cast", ["masked"], ["masked_integer"], name="truncate", to=TensorProto.INT32
        ))
        model.graph.node.insert(7, helper.make_node(
            "Cast", ["masked_integer"], ["masked_restored"], name="restore_type", to=TensorProto.FLOAT
        ))

    path, feeds, expected, directory = _save_mutation(graph_case, mutate)
    validate_onnx_io(path, 1, TensorProto.FLOAT)
    actual = _run(path, feeds)
    np.save(directory / "mutated-output.npy", actual)
    np.testing.assert_array_equal(actual, [[[0.0], [0.0], [0.0], [0.0]]])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_independent_node_reorder_preserves_results(graph_case):
    def mutate(model):
        nodes = list(model.graph.node)
        nodes[0], nodes[1] = nodes[1], nodes[0]
        del model.graph.node[:]
        model.graph.node.extend(nodes)

    path, feeds, expected, _ = _save_mutation(graph_case, mutate)
    validate_onnx_io(path, 1, TensorProto.FLOAT)
    np.testing.assert_array_equal(_run(path, feeds), expected)


def test_same_dtype_cast_preserves_results(graph_case):
    def mutate(model):
        model.graph.node[6].input[0] = "masked_cast"
        model.graph.node.insert(6, helper.make_node(
            "Cast", ["masked"], ["masked_cast"], name="identity_cast", to=TensorProto.FLOAT
        ))

    path, feeds, expected, _ = _save_mutation(graph_case, mutate)
    validate_onnx_io(path, 1, TensorProto.FLOAT)
    np.testing.assert_array_equal(_run(path, feeds), expected)


@pytest.mark.parametrize("contract_error", ["input_name", "input_dtype", "output_dtype", "hidden_size"])
def test_valid_graph_still_must_match_deployment_io_contract(graph_case, contract_error):
    def mutate(model):
        if contract_error == "input_name":
            model.graph.input[0].name = "tokens"
            model.graph.node[0].input[0] = "tokens"
        elif contract_error == "input_dtype":
            model.graph.input[0].type.tensor_type.elem_type = TensorProto.INT64

    path, _, _, _ = _save_mutation(graph_case, mutate)
    # Structural validity alone must not bypass the existing project contract.
    onnx.checker.check_model(str(path), full_check=True)
    expected_type = TensorProto.FLOAT16 if contract_error == "output_dtype" else TensorProto.FLOAT
    hidden_size = 2 if contract_error == "hidden_size" else 1
    messages = {
        "input_name": "unexpected ONNX IO",
        "input_dtype": "must be INT32",
        "output_dtype": "output dtype",
        "hidden_size": "hidden size",
    }
    with pytest.raises(RuntimeError, match=messages[contract_error]):
        validate_onnx_io(path, hidden_size, expected_type)
