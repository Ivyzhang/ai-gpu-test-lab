"""TensorRT input contract and corrupted-engine failure behavior."""
import pytest
import torch

from src.contract import PROFILE_MAX

trt = pytest.importorskip("tensorrt")


@pytest.mark.gpu
@pytest.mark.parametrize("batch,sequence", [(0, 128), (4, PROFILE_MAX[1] + 1)])
def test_out_of_profile_shape_is_rejected(batch, sequence, trt_runner):
    try:
        accepted = trt_runner.context.set_input_shape("input_ids", (batch, sequence))
    except (RuntimeError, ValueError):
        return
    assert accepted is False


@pytest.mark.gpu
def test_wrong_dtype_is_rejected(trt_runner):
    input_ids = torch.zeros((4, 128), device="cuda", dtype=torch.float32)
    attention_mask = torch.ones((4, 128), device="cuda", dtype=torch.int32)
    with pytest.raises(TypeError, match="torch.int32"):
        trt_runner.run_async(input_ids, attention_mask)


@pytest.mark.gpu
def test_out_of_vocabulary_token_is_rejected(trt_runner):
    input_ids = torch.full((1, 1), trt_runner.spec.vocab_size, device="cuda", dtype=torch.int32)
    attention_mask = torch.ones((1, 1), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="input_ids"):
        trt_runner.run_async(input_ids, attention_mask)


@pytest.mark.gpu
def test_invalid_attention_mask_value_is_rejected(trt_runner):
    input_ids = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    attention_mask = torch.tensor([[2]], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="attention_mask"):
        trt_runner.run_async(input_ids, attention_mask)


@pytest.mark.gpu
def test_missing_tensor_name_is_rejected(trt_runner):
    try:
        accepted = trt_runner.context.set_input_shape("not_a_real_tensor_name", (4, 128))
    except (RuntimeError, ValueError):
        return
    assert accepted is False


def test_corrupted_engine_file_is_rejected(tmp_path):
    from src.trt_runner import TrtRunner

    bad_engine = tmp_path / "corrupted.plan"
    bad_engine.write_bytes(b"not a real engine")
    with pytest.raises(RuntimeError, match="deserialize"):
        TrtRunner(str(bad_engine))
