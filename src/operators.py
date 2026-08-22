"""Reference (PyTorch CPU FP32) / candidate (PyTorch CUDA) operator interfaces."""
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


class OperatorUnderTest:
    """Unified reference(CPU)/candidate(CUDA) interface shared by the test matrix."""

    name: str

    def reference(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def candidate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def validate_input(x: Tensor) -> None:
        if not x.is_floating_point():
            raise TypeError("operator expects a floating-point input tensor")


class MatMulOp(OperatorUnderTest):
    name = "matmul"

    def __init__(self, k: int, n: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(k, n, generator=g)

    def reference(self, x, **kwargs):
        return x.cpu().float() @ self.weight.float()

    def candidate(self, x, dtype=torch.float16, **kwargs):
        self.validate_input(x)
        w = self.weight.to(device="cuda", dtype=dtype)
        return x.to(device="cuda", dtype=dtype) @ w


class LayerNormOp(OperatorUnderTest):
    name = "layernorm"

    def __init__(self, hidden: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(hidden, generator=g)
        self.bias = torch.randn(hidden, generator=g)

    def reference(self, x, **kwargs):
        return F.layer_norm(x.cpu().float(), (x.shape[-1],), self.weight.float(), self.bias.float())

    def candidate(self, x, dtype=torch.float16, **kwargs):
        self.validate_input(x)
        w = self.weight.to(device="cuda", dtype=dtype)
        b = self.bias.to(device="cuda", dtype=dtype)
        return F.layer_norm(x.to(device="cuda", dtype=dtype), (x.shape[-1],), w, b)


class SoftmaxOp(OperatorUnderTest):
    name = "softmax"

    def reference(self, x, **kwargs):
        return F.softmax(x.cpu().float(), dim=-1)

    def candidate(self, x, dtype=torch.float16, **kwargs):
        self.validate_input(x)
        return F.softmax(x.to(device="cuda", dtype=dtype), dim=-1)


class GeluOp(OperatorUnderTest):
    name = "gelu"

    def reference(self, x, **kwargs):
        return F.gelu(x.cpu().float())

    def candidate(self, x, dtype=torch.float16, **kwargs):
        self.validate_input(x)
        return F.gelu(x.to(device="cuda", dtype=dtype))


# matmul/layernorm own parameters sized to the input's last dim, so they are
# built per-case (hidden=case.shape[-1]) instead of once with a fixed size;
# the default keeps zero-arg call sites (e.g. `OPERATORS["gelu"]()`) working.
OPERATORS = {
    "matmul": lambda hidden=1024: MatMulOp(k=hidden, n=hidden),
    "layernorm": lambda hidden=1024: LayerNormOp(hidden=hidden),
    "softmax": lambda hidden=1024: SoftmaxOp(),
    "gelu": lambda hidden=1024: GeluOp(),
}
