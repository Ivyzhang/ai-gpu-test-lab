"""Compile a selected workload ONNX graph into a TensorRT FP16 Engine."""
import argparse
import hashlib

import tensorrt as trt

from src.contract import INPUT_NAMES, OUTPUT_NAME
from src.workloads import WORKLOADS, WORKLOAD_NAMES

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path: str, engine_path: str, workload_name: str = "distilbert-base-uncased") -> dict:
    if workload_name not in WORKLOADS:
        raise ValueError(f"unknown workload: {workload_name}; choose from {WORKLOAD_NAMES}")
    spec = WORKLOADS[workload_name]
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 0

    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        network_flags |= 1 << int(
            trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED
        )

    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as handle:
        if not parser.parse(handle.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errors}")

    actual_inputs = {network.get_input(index).name for index in range(network.num_inputs)}
    actual_outputs = {network.get_output(index).name for index in range(network.num_outputs)}
    if actual_inputs != set(INPUT_NAMES) or actual_outputs != {OUTPUT_NAME}:
        raise RuntimeError(f"unexpected IO: inputs={actual_inputs}, outputs={actual_outputs}")
    input_tensors = {
        network.get_input(index).name: network.get_input(index)
        for index in range(network.num_inputs)
    }
    for name in INPUT_NAMES:
        if input_tensors[name].dtype != trt.DataType.INT32:
            raise TypeError(f"ONNX input {name} must be INT32")
    output_tensor = network.get_output(0)
    if output_tensor.dtype != trt.DataType.HALF:
        raise TypeError("TensorRT deployment graph output must be HALF")

    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    for name in INPUT_NAMES:
        profile.set_shape(
            name,
            min=spec.profile_min,
            opt=spec.profile_opt,
            max=spec.profile_max,
        )
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")
    with open(engine_path, "wb") as handle:
        handle.write(serialized_engine)

    return {
        "workload_name": workload_name,
        "engine_sha256": hashlib.sha256(serialized_engine).hexdigest(),
        "profile": {"min": spec.profile_min, "opt": spec.profile_opt, "max": spec.profile_max},
        "output_hidden": spec.hidden,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="model.onnx")
    parser.add_argument("--engine", default="model.plan")
    parser.add_argument("--workload", choices=WORKLOAD_NAMES, default="distilbert-base-uncased")
    args = parser.parse_args()
    print(build_engine(args.onnx, args.engine, args.workload))

