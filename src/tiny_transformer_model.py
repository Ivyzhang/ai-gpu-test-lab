"""Original deterministic TinyTransformer workload retained for comparison."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int = 32000, hidden: int = 256, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.linear1 = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)

    def forward_fp32(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids.long())
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        x = x * mask
        hidden = self.norm1(x + F.gelu(self.linear1(x)))
        return self.norm2(hidden + F.softmax(self.linear2(hidden), dim=-1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.forward_fp32(input_ids, attention_mask).to(torch.float16)
