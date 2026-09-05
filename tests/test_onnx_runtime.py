"""ONNX Runtime separates export errors from TensorRT errors per workload."""
import pytest
import torch

from src.contract import SUPPORTED_WORKLOAD_SHAPES
from src.workloads import WORKLOADS, make_inputs, reference_forward


@pytest.mark.parametrize("batch,sequence", list(SUPPORTED_WORKLOAD_SHAPES))
def test_onnx_runtime_matches_pytorch(batch, sequence, workload_name, ort_session):
    spec = WORKLOADS[workload_name]
    model = spec.create_model().eval()
    input_ids, attention_mask = make_inputs(spec, batch, sequence, seed=batch * 1000 + sequence)

    with torch.no_grad():
        reference = reference_forward(model, input_ids, attention_mask)
    actual = torch.from_numpy(
        ort_session.run(
            ["hidden_states"],
            {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()},
        )[0]
    ).float()

    assert actual.shape == (batch, sequence, spec.hidden)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=5e-3, rtol=5e-3)


def test_tokenizer_output_matches_distilbert_contract(workload_name, ort_session):
    if workload_name != "distilbert-base-uncased":
        pytest.skip("tokenizer-specific case")
    from src.distilbert_base_uncased_model import load_tokenizer, MODEL_REVISION

    tokenizer = load_tokenizer(revision=MODEL_REVISION)
    encoded = tokenizer(
        ["GPU inference quality gate", "TensorRT and ONNX"],
        padding="max_length",
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(torch.int32)
    attention_mask = encoded["attention_mask"].to(torch.int32)
    output = torch.from_numpy(
        ort_session.run(
            ["hidden_states"],
            {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()},
        )[0]
    )
    assert input_ids.shape == attention_mask.shape == (2, 16)
    assert int(input_ids.min()) >= 0
    assert int(input_ids.max()) < WORKLOADS[workload_name].vocab_size
    assert torch.isfinite(output).all()
