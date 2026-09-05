"""Model-specific contracts shared by export, runtime and parameterized tests."""
from __future__ import annotations

from dataclasses import dataclass
import os

from src.contract import PROFILE_MAX, PROFILE_MIN, PROFILE_OPT
from src.distilbert_base_uncased_model import DistilBertEncoder
from src.tiny_transformer_model import TinyTransformerEncoder


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    model_name: str
    revision: str
    hidden: int
    vocab_size: int
    model_factory: object

    @property
    def profile_min(self):
        return PROFILE_MIN

    @property
    def profile_opt(self):
        return PROFILE_OPT

    @property
    def profile_max(self):
        return PROFILE_MAX

    def create_model(self):
        return self.model_factory()


DISTILBERT_REVISION = os.environ.get("DISTILBERT_REVISION", "main")

WORKLOADS = {
    "tiny-transformer": WorkloadSpec(
        name="tiny-transformer",
        model_name="TinyTransformerEncoder",
        revision="local-seed-0",
        hidden=256,
        vocab_size=32000,
        model_factory=lambda: TinyTransformerEncoder(),
    ),
    "distilbert-base-uncased": WorkloadSpec(
        name="distilbert-base-uncased",
        model_name="distilbert-base-uncased",
        revision=DISTILBERT_REVISION,
        hidden=768,
        vocab_size=30522,
        model_factory=lambda: DistilBertEncoder(revision=DISTILBERT_REVISION),
    ),
}

WORKLOAD_NAMES = tuple(WORKLOADS)


def make_inputs(spec: WorkloadSpec, batch: int, sequence: int, seed: int = 0, device=None):
    import torch

    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0, spec.vocab_size, (batch, sequence), generator=generator, dtype=torch.int32
    )
    attention_mask = torch.ones((batch, sequence), dtype=torch.int32)
    if sequence > 1:
        attention_mask[:, sequence // 2 :] = 0
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def reference_forward(model, input_ids, attention_mask):
    if hasattr(model, "forward_fp32"):
        return model.forward_fp32(input_ids, attention_mask).float()
    return model(input_ids, attention_mask).float()
