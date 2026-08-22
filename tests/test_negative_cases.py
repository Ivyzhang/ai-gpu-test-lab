"""Profile-outside shapes, wrong dtype, missing tensor names and corrupted engines must fail predictably."""
import pytest
import torch

from src.contract import PROFILE_MAX, VOCAB_SIZE

trt = pytest.importorskip("tensorrt")

OUT_OF_PROFILE_SHAPES = [(0, 128), (4, PROFILE_MAX[1] + 1)]


@pytest.mark.gpu
@pytest.mark.parametrize("batch,seq", OUT_OF_PROFILE_SHAPES)
def test_out_of_profile_shape_is_rejected(batch, seq, trt_runner):
    with pytest.raises((RuntimeError, ValueError)):
        trt_runner.context.set_input_shape("input_ids", (batch, seq))


@pytest.mark.gpu
def test_wrong_dtype_input_is_rejected(trt_runner):
    batch, seq = 4, 128
    input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq), device="cuda", dtype=torch.float32)  # wrong dtype
    attention_mask = torch.ones(batch, seq, device="cuda", dtype=torch.int32)
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        trt_runner.run(input_ids, attention_mask)


@pytest.mark.gpu
def test_missing_tensor_name_raises(trt_runner):
    with pytest.raises((RuntimeError, ValueError)):
        trt_runner.context.set_input_shape("not_a_real_tensor_name", (4, 128))


def test_corrupted_engine_file_deserializes_to_none(tmp_path):
    """TensorRT returns None (not an exception) for a corrupted plan; callers must check explicitly."""
    bad_engine = tmp_path / "corrupted.plan"
    bad_engine.write_bytes(b"not a real engine")

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(bad_engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine is None


def test_trt_runner_raises_on_corrupted_engine(tmp_path):
    from src.trt_runner import TrtRunner

    bad_engine = tmp_path / "corrupted.plan"
    bad_engine.write_bytes(b"not a real engine")
    with pytest.raises(RuntimeError):
        TrtRunner(str(bad_engine))
