"""Fixed TinyTransformerEncoder workload: Embedding -> Linear/GELU/LayerNorm -> Linear/Softmax/LayerNorm."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.contract import HIDDEN, VOCAB_SIZE


class TinyTransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden: int = HIDDEN, seed: int = 0):
        super().__init__()
        self.hidden = hidden
        g = torch.Generator().manual_seed(seed)
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.linear1 = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        # fixed init instead of PyTorch's default RNG state, so weights are reproducible across processes
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids.long())
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        x = x * mask
        h = self.norm1(x + F.gelu(self.linear1(x)))
        h = self.norm2(h + F.softmax(self.linear2(h), dim=-1))
        return h
