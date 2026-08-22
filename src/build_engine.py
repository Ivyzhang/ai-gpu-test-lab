"""Compile model.onnx into an FP16 TensorRT engine using the fixed optimization profile."""
import argparse
import hashlib

import tensorrt as trt

from src.contract import INPUT_NAMES, PROFILE_MAX, PROFILE_MIN, PROFILE_OPT

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path: str, engine_path: str) -> dict:
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    for name in INPUT_NAMES:
        profile.set_shape(name, min=PROFILE_MIN, opt=PROFILE_OPT, max=PROFILE_MAX)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    engine_hash = hashlib.sha256(serialized_engine).hexdigest()
    return {
        "engine_sha256": engine_hash,
        "profile": {"min": PROFILE_MIN, "opt": PROFILE_OPT, "max": PROFILE_MAX},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="model.onnx")
    parser.add_argument("--engine", default="model.plan")
    args = parser.parse_args()
    print(build_engine(args.onnx, args.engine))
