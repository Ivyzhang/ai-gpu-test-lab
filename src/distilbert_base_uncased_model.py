"""DistilBERT workload with explicit local/remote model source handling."""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = os.environ.get("DISTILBERT_REVISION", "main")


def resolve_model_source(model_dir: str | None = None, model_name: str = MODEL_NAME) -> str:
    source = model_dir or os.environ.get("DISTILBERT_LOCAL_PATH") or model_name
    if not source:
        raise ValueError("set DISTILBERT_LOCAL_PATH or provide model_dir/model_name")
    return source


class DistilBertEncoder(nn.Module):
    """Expose DistilBERT last_hidden_state through the project contract."""

    def __init__(
        self,
        model_dir: str | None = None,
        model_name: str = MODEL_NAME,
        revision: str = MODEL_REVISION,
        local_files_only: bool | None = None,
    ):
        super().__init__()
        source = resolve_model_source(model_dir, model_name)
        if local_files_only is None:
            local_files_only = bool(model_dir or os.environ.get("DISTILBERT_LOCAL_PATH"))

        self.model_dir = source
        self.model_name = model_name
        self.revision = revision
        self.backbone = AutoModel.from_pretrained(
            source,
            revision=None if local_files_only else revision,
            local_files_only=local_files_only,
        )
        self.backbone.eval()

    def forward_fp32(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
        )
        return outputs.last_hidden_state.float()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.forward_fp32(input_ids, attention_mask).to(torch.float16)


def load_tokenizer(
    model_dir: str | None = None,
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
    local_files_only: bool | None = None,
):
    source = resolve_model_source(model_dir, model_name)
    if local_files_only is None:
        local_files_only = bool(model_dir or os.environ.get("DISTILBERT_LOCAL_PATH"))
    return AutoTokenizer.from_pretrained(
        source,
        revision=None if local_files_only else revision,
        local_files_only=local_files_only,
    )

