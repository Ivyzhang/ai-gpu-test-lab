"""Session-scoped ONNX export + TensorRT engine build shared by TensorRT tests."""
import os
from pathlib import Path

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()

try:
    import tensorrt  # noqa: F401

    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False


@pytest.fixture(scope="session")
def engine_paths(tmp_path_factory):
    """Export ONNX and build the FP16 TensorRT engine once per test session."""
    if not (HAS_CUDA and HAS_TENSORRT):
        pytest.skip("requires a CUDA GPU with TensorRT installed")

    from src.build_engine import build_engine
    from src.export_onnx import export

    report_dir = os.environ.get("REPORT_DIR")
    if report_dir:
        workdir = Path(report_dir) / "artifacts"
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        workdir = tmp_path_factory.mktemp("trt_engine")
    onnx_path = workdir / "model.onnx"
    engine_path = workdir / "model.plan"

    export_info = export(str(onnx_path))
    build_info = build_engine(str(onnx_path), str(engine_path))
    return {"onnx": onnx_path, "engine": engine_path, **export_info, **build_info}


@pytest.fixture
def trt_runner(engine_paths):
    from src.trt_runner import TrtRunner

    return TrtRunner(str(engine_paths["engine"]))
