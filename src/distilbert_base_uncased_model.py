"""DistilBERT workload used by the PyTorch, ONNX and TensorRT paths."""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = os.environ.get("DISTILBERT_REVISION", "main")


class DistilBertEncoder(nn.Module):
    """Expose DistilBERT last_hidden_state with the project's tensor contract."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        revision: str = MODEL_REVISION,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.revision = revision
        self.backbone = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
        )
        self.backbone.eval()

    def forward_fp32(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
        )
        return outputs.last_hidden_state

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.forward_fp32(input_ids, attention_mask).to(torch.float16)


def load_tokenizer(
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
    local_files_only: bool = False,
):
    return AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
    )
