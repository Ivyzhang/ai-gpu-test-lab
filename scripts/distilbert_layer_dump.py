"""DistilBERT 逐层 dump。定位用，不作为发布门禁。

三条取层路径：

1. 参考层：PyTorch forward hook。层就是 embeddings 和 6 个 transformer block，
   模块都在，输出可以直接拿。
2. ORT 实际层：ORT 只返回 graph output。先用 shape inference 给中间 tensor
   补上名字和类型，再把要看的那一层追加成输出，session 才能吐出来。
3. TensorRT 实际层：编译时中间 tensor 会被融合掉。必须在 build 之前把该层
   标成 ONNX 输出，重新 build。一次只标一个切分点，标多了融合就没了，
   你看的不是原来那张图。Engine 建好之后，按这个输出名取，和取最终
   hidden_states 是同一个动作。

比较只在 attention_mask == 1 的位置上做。padding 没有定义，不参与。
"""
from __future__ import annotations

import torch
import onnx
from onnx import shape_inference

from src.distilbert_base_uncased_model import DistilBertEncoder

ATOL = 2.5e-2
RTOL = 1e-2
LAYER_NAMES = ["embeddings"] + [f"transformer.layer.{i}" for i in range(6)]


# ---------- 层清单：只有这部分跟 DistilBERT 走 ----------

def pytorch_layers(encoder: DistilBertEncoder) -> list[tuple[str, torch.nn.Module]]:
    backbone = encoder.backbone
    layers = [("embeddings", backbone.embeddings)]
    for index, block in enumerate(backbone.transformer.layer):
        layers.append((f"transformer.layer.{index}", block))
    return layers


def hidden_from_output(output) -> torch.Tensor:
    """Embedding 返回 Tensor。block 返回 (hidden_state,)，取第 0 个。"""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"不认识的层输出: {type(output)}")


# ---------- 参考层：PyTorch ----------

def dump_reference(encoder: DistilBertEncoder, input_ids: torch.Tensor,
                   attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    dumped: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            # 立刻 detach 并拷到 CPU。留在图里会被后续层改写。
            dumped[name] = hidden_from_output(output).detach().float().cpu()
        return hook

    for name, module in pytorch_layers(encoder):
        handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        dumped["__final__"] = encoder.forward_fp32(input_ids, attention_mask).detach().float().cpu()

    for handle in handles:
        handle.remove()
    return dumped


# ---------- 实际层：ONNX，先标输出，ORT 和 TRT 共用这一步 ----------

def intermediate_names(model_path: str) -> list[str]:
    """导出后的中间 tensor 名不稳定，先列出来再选，不要手写。"""
    graph = shape_inference.infer_shapes(onnx.load(model_path))
    return [value.name for value in graph.graph.value_info]


def mark_layer(model_path: str, out_path: str, tensor_name: str) -> str:
    """把一个中间 tensor 追加成 graph output。返回实际写入的名字。"""
    graph = shape_inference.infer_shapes(onnx.load(model_path))
    existing = {output.name for output in graph.graph.output}
    if tensor_name not in existing:
        value_info = {value.name: value for value in graph.graph.value_info}
        if tensor_name not in value_info:
            raise KeyError(f"{tensor_name} 不在这张图里。先用 intermediate_names 看候选")
        graph.graph.output.append(value_info[tensor_name])
    onnx.save(graph, out_path)
    return tensor_name


def dump_ort(session, tensor_name: str, input_ids: torch.Tensor,
             attention_mask: torch.Tensor) -> torch.Tensor:
    """session 必须用 mark_layer 之后的图来建。"""
    produced = {output.name for output in session.get_outputs()}
    if tensor_name not in produced:
        raise KeyError(f"session 没有 {tensor_name}，确认 mark_layer 用在了建 session 之前")
    values = session.run(
        [tensor_name],
        {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()},
    )
    return torch.from_numpy(values[0]).float()


def dump_trt(runner, tensor_name: str, input_ids: torch.Tensor,
             attention_mask: torch.Tensor) -> torch.Tensor:
    """runner 的 engine 必须用 mark_layer 之后的图重新 build。

    标成输出之后，这一层和 hidden_states 一样，是 Engine 的一个输出，
    用同一套 set_input_shape / set_tensor_address / execute_async_v3 去取。
    没有标成输出的层，Engine 里不存在，不能在运行时临时要。
    """
    if tensor_name not in set(runner.engine):
        raise KeyError(f"engine 没有输出 {tensor_name}。回到编译前 mark_layer，再 build")
    runner.run(input_ids, attention_mask)
    # 具体的地址绑定走 TrtRunner。这里只表明取数的位置：按输出名，不按层号。
    shape = tuple(runner.context.get_tensor_shape(tensor_name))
    output = torch.empty(shape, device=input_ids.device, dtype=torch.float16)
    runner.context.set_tensor_address(tensor_name, output.data_ptr())
    return output


# ---------- 比较：跟模型无关 ----------

def compare(candidate: torch.Tensor, reference: torch.Tensor,
            attention_mask: torch.Tensor) -> dict:
    mask = attention_mask.to(torch.bool).unsqueeze(-1).expand_as(reference)
    diff = (candidate.float() - reference.float()).abs()
    valid = diff[mask]
    limit = ATOL + RTOL * reference.float().abs()[mask]
    return {
        "finite": bool(torch.isfinite(candidate).all()),
        "max_abs": valid.max().item() if valid.numel() else 0.0,
        "violations": int((valid > limit).sum()),
    }


def first_bad_layer(reference: dict[str, torch.Tensor],
                    candidate: dict[str, torch.Tensor],
                    attention_mask: torch.Tensor) -> str | None:
    for name in LAYER_NAMES:
        if name not in candidate:
            continue
        result = compare(candidate[name], reference[name], attention_mask)
        if not result["finite"] or result["violations"]:
            return name
    return None


if __name__ == "__main__":
    encoder = DistilBertEncoder().eval()
    input_ids = torch.randint(0, 30522, (2, 16), dtype=torch.int32)
    attention_mask = torch.ones(2, 16, dtype=torch.int32)
    attention_mask[:, 10:] = 0

    reference = dump_reference(encoder, input_ids, attention_mask)
    for name in LAYER_NAMES:
        print("reference", name, tuple(reference[name].shape))
