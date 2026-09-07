"""Shared parameterized artifacts for both inference workloads."""
import os
from pathlib import Path

import pytest
import torch

from src.workloads import WORKLOADS, WORKLOAD_NAMES

HAS_CUDA = torch.cuda.is_available()

try:
    import tensorrt  # noqa: F401
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False


@pytest.fixture(scope="session", params=WORKLOAD_NAMES, ids=WORKLOAD_NAMES)
def workload_name(request):
    return request.param


def _artifact_dir(tmp_path_factory) -> Path:
    report_dir = os.environ.get("REPORT_DIR")
    if report_dir:
        path = Path(report_dir) / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path_factory.mktemp("model_artifacts")


@pytest.fixture(scope="session")
def onnx_paths(tmp_path_factory, workload_name):
    from src.export_onnx import export_workload

    workdir = _artifact_dir(tmp_path_factory)
    return export_workload(workdir, workload_name)


@pytest.fixture(scope="session")
def engine_paths(onnx_paths):
    if not (HAS_CUDA and HAS_TENSORRT):
        pytest.skip("requires a CUDA GPU with TensorRT installed")

    from src.build_engine import build_engine

    workload_name = onnx_paths["workload_name"]
    engine_path = onnx_paths["fp16_onnx"].with_suffix(".plan")
    return {
        **onnx_paths,
        "engine": engine_path,
        **build_engine(str(onnx_paths["fp16_onnx"]), str(engine_path), workload_name),
    }


@pytest.fixture(scope="session")
def ort_session(onnx_paths):
    ort = pytest.importorskip("onnxruntime")
    return ort.InferenceSession(str(onnx_paths["fp32_onnx"]), providers=["CPUExecutionProvider"])


@pytest.fixture
def trt_runner(engine_paths):
    from src.trt_runner import TrtRunner

    runner = TrtRunner(str(engine_paths["engine"]), engine_paths["workload_name"])
    try:
        yield runner
    finally:
        runner.close()

