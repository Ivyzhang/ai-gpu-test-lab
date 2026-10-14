"""逐层 dump：同一输入下，把 PyTorch 和 ONNX Runtime 的中间张量逐层比。

这是定位工具，不是发布门禁。门禁只看最终 hidden_states。
TensorRT 编译时会融掉中间张量，要看某一层必须在编译前把该张量标成
ONNX 输出再重新 build。这里先做 PyTorch 对 ORT，TRT 用同一个 mark 方法。
"""
from __future__ import annotations

import torch
import onnx

from src.tiny_transformer_model import TinyTransformerEncoder


def layer_names(model: torch.nn.Module) -> list[str]:
    # 叶子模块才是可对齐的一层。容器本身的输出等于最后一层，不重复记。
    return [name for name, module in model.named_modules() if name and not list(module.children())]


def dump_pytorch(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    dumped: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            # hook 里拿到的是计算图上的 tensor，先 detach 再拷走，避免后面被原地改掉。
            dumped[name] = output.detach()
        return hook

    for name, module in model.named_modules():
        if name and not list(module.children()):
            handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        final = model.forward_fp32(input_ids, attention_mask)
    dumped["__final__"] = final.detach()

    for handle in handles:
        handle.remove()
    return dumped


def mark_onnx_intermediates(model_path: str, out_path: str, tensor_names: list[str]) -> None:
    """把中间 tensor 追加成 graph output。ORT 只会返回 graph output。"""
    graph = onnx.load(model_path)
    existing = {output.name for output in graph.graph.output}
    value_info = {value.name: value for value in graph.graph.value_info}

    for name in tensor_names:
        if name in existing:
            continue
        if name not in value_info:
            raise KeyError(f"{name} 不是图里的中间 tensor，先确认导出时的节点名")
        graph.graph.output.append(value_info[name])

    onnx.save(graph, out_path)


def dump_ort(session, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    output_names = [output.name for output in session.get_outputs()]
    values = session.run(
        output_names,
        {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()},
    )
    return {name: torch.from_numpy(value) for name, value in zip(output_names, values)}


def valid_mask(attention_mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    # attention_mask: [B, S]，中间张量: [B, S, H]。只保留 mask 为 1 的位置。
    mask = attention_mask.to(like.device)
    if like.ndim == 3:
        mask = mask.unsqueeze(-1)
    return mask.to(dtype=torch.bool).expand_as(like)


def layer_metrics(candidate: torch.Tensor, reference: torch.Tensor, attention_mask: torch.Tensor) -> dict:
    mask = valid_mask(attention_mask, reference)
    diff = (candidate.float() - reference.float()).abs()
    valid = diff[mask]
    return {
        "finite": bool(torch.isfinite(candidate).all()),
        "max_abs": valid.max().item() if valid.numel() else 0.0,
        "mean_abs": valid.mean().item() if valid.numel() else 0.0,
        "violations": int((valid > (5e-3 + 5e-3 * reference.float().abs()[mask])).sum()),
    }


def first_diverging_layer(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor],
                          attention_mask: torch.Tensor, names: list[str]) -> tuple[str | None, dict]:
    """按层顺序找第一个越线的层。层数多时这里改成按层号二分。"""
    report = {}
    for name in names:
        if name not in candidate:
            continue
        metrics = layer_metrics(candidate[name], reference[name], attention_mask)
        report[name] = metrics
        if not metrics["finite"] or metrics["violations"] > 0:
            return name, report
    return None, report


if __name__ == "__main__":
    model = TinyTransformerEncoder().eval()
    input_ids = torch.randint(0, 32000, (2, 8), dtype=torch.int32)
    attention_mask = torch.ones(2, 8, dtype=torch.int32)
    attention_mask[:, 5:] = 0

    reference = dump_pytorch(model, input_ids, attention_mask)
    for name in ["embedding", "linear1", "norm1", "linear2", "norm2"]:
        print(name, tuple(reference[name].shape))
