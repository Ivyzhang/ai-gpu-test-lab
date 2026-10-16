# RTX 4090D 24GB：DistilBERT 与 Qwen 部署/量化测试

本目录提供两条独立的模型测试路径：

- **DistilBERT**：Encoder 模型，测试 `last_hidden_state` 数值一致性、输入边界、FP16/INT8/NF4 量化和 1/2/4/8 并发。
- **Qwen2.5-0.5B-Instruct**：Decoder-only 生成模型，通过 vLLM OpenAI 兼容服务测试流式输出、TTFT、E2E、TPOT 估计、吞吐、并发和固定任务质量。

DistilBERT 没有自回归生成过程，所以不报告 TTFT/TPOT；它使用完整请求延迟。Qwen 才使用 TTFT/TPOT。两者的量化结果、基线和报告必须分开。

## 1. Docker 与 GPU 检查

宿主机先确认 NVIDIA 驱动和 Container Toolkit 可用：

```bash
nvidia-smi
nvidia-ctk --version

docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

如果第三条命令失败，先修复宿主机驱动、NVIDIA Container Toolkit、Docker GPU runtime；模型测试还没有开始。4090D 是单卡 24GB，本方案所有服务默认单 GPU，`CUDA_VISIBLE_DEVICES=0`，不启用 Tensor Parallel。

构建测试镜像：

```bash
cd /workspace/ai-gpu-test-lab

# 用团队批准的 vLLM 版本替换下面的占位符；不要用 latest 做发布基线
docker build -f tests/model_deployment/Dockerfile \
  --build-arg VLLM_SPEC='vllm==<approved-version>' \
  --build-arg BNB_SPEC='bitsandbytes==<approved-version>' \
  -t ai-gpu-quant-test:py311 .
```

这个 Dockerfile 安装 vLLM、bitsandbytes、Transformers、Accelerate、Hugging Face Hub 和 HTTP 测试依赖。若只做探索，可把两个 build arg 改成 `vllm`/`bitsandbytes`；正式测试必须填写已经在 4090D + 当前驱动上验证过的版本。vLLM 容器建议使用 `--ipc=host` 或足够大的 `--shm-size`，避免多进程共享内存不足。vLLM 的 CUDA/PyTorch 兼容性以实际安装结果为准：

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace" \
  -v /models:/models \
  ai-gpu-quant-test:py311 \
  bash -lc 'python - <<"PY"
import torch, vllm, bitsandbytes
print("vLLM", vllm.__version__)
print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("bitsandbytes", bitsandbytes.__version__)
assert torch.cuda.is_available()
PY'
```

## 2. 下载并固定模型

下载脚本会获取 Hugging Face 的 immutable commit SHA，然后下载原始权重、配置、tokenizer 和 chat template，并写入 `snapshot-manifest.json`。它不会把量化权重混入原始模型目录。

网络可用的 GPU 主机上运行：

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace" \
  -v /models:/models \
  ai-gpu-quant-test:py311 \
  bash -lc 'cd /workspace && \
    python -m tests.model_deployment.download_models \
      --root /models \
      --distilbert-revision main \
      --qwen-revision main'
```

正式发布不要使用浮动的 `main`。先在有网络的机器取得并审核提交 SHA，然后用 SHA：

```bash
# 示例格式；替换成实际审核过的 commit SHA
DISTILBERT_SHA=<approved-distilbert-commit>
QWEN_SHA=<approved-qwen-commit>

docker run --rm --gpus all \
  -v "$PWD:/workspace" -v /models:/models \
  ai-gpu-quant-test:py311 bash -lc "cd /workspace && \
  python -m tests.model_deployment.download_models \
    --root /models \
    --distilbert-revision '$DISTILBERT_SHA' \
    --qwen-revision '$QWEN_SHA'"
```

模型目录应包含：

```text
/models/distilbert-base-uncased/
/models/Qwen2.5-0.5B-Instruct/
```

如果目录已经存在，下载脚本会停止，避免把不同 revision 混合到同一目录。检查身份：

```bash
cat /models/distilbert-base-uncased/snapshot-manifest.json
cat /models/Qwen2.5-0.5B-Instruct/snapshot-manifest.json
```

## 3. DistilBERT：先建立 FP32 参考

DistilBERT 服务是 HTTP Encoder 服务，不使用 `/v1/chat/completions`。它提供：

```text
GET  /health
POST /encode  {"texts": ["..."]}
```

每次只启动一种精度/量化模式，避免 24GB 显存争用。终端 A 启动 FP32：

```bash
docker run --rm --gpus all --ipc=host --name distilbert-fp32 \
  -v "$PWD:/workspace" -v /models:/models \
  -p 8001:8001 \
  ai-gpu-quant-test:py311 \
  bash -lc 'cd /workspace && \
    python -m tests.model_deployment.distilbert_server \
      --model-dir /models/distilbert-base-uncased \
      --mode fp32 --port 8001'
```

看到日志里的 readiness inference 和模型信息后，在终端 B 捕获 FP32 参考：

```bash
mkdir -p reports/distilbert/fp32-reference

docker run --rm --network host \
  -v "$PWD:/workspace" \
  ai-gpu-quant-test:py311 \
  bash -lc 'cd /workspace && \
    python -m tests.model_deployment.distilbert_test \
      --url http://127.0.0.1:8001 \
      --output reports/distilbert/fp32-reference \
      --requests 32'
```

该次结果应为 `REFERENCE_CAPTURED`，生成：

```text
reports/distilbert/fp32-reference/references.json
reports/distilbert/fp32-reference/summary.json
```

FP32 参考捕获完成后停止服务：

```bash
docker stop distilbert-fp32
```

参考必须包含模型 hash、输入数据 hash 和完整输出；不能用不同 revision、不同 tokenizer 或不同输入集作为 reference。

## 4. DistilBERT：FP16、INT8、NF4 测试

每种模式都启动一个新容器，然后用同一份 `references.json`。以下示例测试 FP16：

```bash
docker run --rm --gpus all --ipc=host --name distilbert-fp16 \
  -v "$PWD:/workspace" -v /models:/models -p 8001:8001 \
  ai-gpu-quant-test:py311 \
  bash -lc 'cd /workspace && \
    python -m tests.model_deployment.distilbert_server \
      --model-dir /models/distilbert-base-uncased \
      --mode fp16 --port 8001'
```

另一个终端：

```bash
mkdir -p reports/distilbert/fp16

docker run --rm --network host \
  -v "$PWD:/workspace" \
  ai-gpu-quant-test:py311 \
  bash -lc 'cd /workspace && \
    python -m tests.model_deployment.distilbert_test \
      --url http://127.0.0.1:8001 \
      --reference reports/distilbert/fp32-reference/references.json \
      --output reports/distilbert/fp16 \
      --requests 32 --atol 2e-2 --rtol 2e-2'
```

FP16 结束后停止服务，再测试 INT8：

```bash
docker stop distilbert-fp16

docker run --rm --gpus all --ipc=host --name distilbert-int8 \
  -v "$PWD:/workspace" -v /models:/models -p 8001:8001 \
  ai-gpu-quant-test:py311 \
  bash -lc 'cd /workspace && \
    python -m tests.model_deployment.distilbert_server \
      --model-dir /models/distilbert-base-uncased \
      --mode int8 --port 8001'
```

```bash
mkdir -p reports/distilbert/int8

docker run --rm --network host \
  -v "$PWD:/workspace" \
  ai-gpu-quant-test:py311 \
  bash -lc 'cd /workspace && \
    python -m tests.model_deployment.distilbert_test \
      --url http://127.0.0.1:8001 \
      --reference reports/distilbert/fp32-reference/references.json \
      --output reports/distilbert/int8 \
      --requests 32 --atol 1e-1 --rtol 5e-2'
```

NF4 是 4-bit 权重、FP16 计算的 bitsandbytes 路径：

```bash
docker stop distilbert-int8

docker run --rm --gpus all --ipc=host --name distilbert-nf4 \
  -v "$PWD:/workspace" -v /models:/models -p 8001:8001 \
  ai-gpu-quant-test:py311 \
  bash -lc 'cd /workspace && \
    python -m tests.model_deployment.distilbert_server \
      --model-dir /models/distilbert-base-uncased \
      --mode nf4 --port 8001'
```

```bash
mkdir -p reports/distilbert/nf4

docker run --rm --network host \
  -v "$PWD:/workspace" \
  ai-gpu-quant-test:py311 \
  bash -lc 'cd /workspace && \
    python -m tests.model_deployment.distilbert_test \
      --url http://127.0.0.1:8001 \
      --reference reports/distilbert/fp32-reference/references.json \
      --output reports/distilbert/nf4 \
      --requests 32 --atol 2e-1 --rtol 1e-1'
```

当前 DistilBERT 测试会检查：

- 模型 hash、tokenizer 后输入 ID、attention mask 一致；
- 输出 shape `[batch, sequence, 768]`；
- 输出无 NaN/Inf；
- masked token 仍按实际输出比较，不能假设最终 hidden state 为 0；
- max sequence 边界、空输入、过长输入和 batch>8 被拒绝；
- 非法请求后的正常请求；
- 1/2/4/8 并发下响应 ID、结果归属、成功率和 E2E p50/p95；
- 量化模块确实替换了 Linear 层，并记录显存峰值。

`--atol/--rtol` 必须由校准实验和业务效果确定。示例阈值只是起始实验值，不是发布标准。FP32 reference 只是数值 reference；如果 DistilBERT 用于分类/NER，还要增加真实标注集上的 accuracy、F1、召回率等任务指标。

## 5. Qwen：vLLM FP16 基准

Qwen 由 vLLM OpenAI 兼容服务启动。终端 A：

```bash
docker run --rm --gpus all --ipc=host --name qwen-vllm-fp16 \
  -v "$PWD:/workspace" -v /models:/models -p 8000:8000 \
  ai-gpu-quant-test:py311 \
  bash -lc 'cd /workspace && \
    python -m tests.llm_e2e.serve \
      --model-dir /models/Qwen2.5-0.5B-Instruct \
      --cuda-device 0 --dtype half \
      --output reports/qwen-vllm-fp16-server'
```

等待 `READY` 后，在终端 B 运行原有端到端矩阵：

```bash
cp tests/llm_e2e/config.example.json tests/llm_e2e/config.gpu.json

# 第一次先探索，SLA 的三个字段保持 null
mkdir -p reports/qwen-vllm-fp16-test

docker run --rm --network host \
  -v "$PWD:/workspace" \
  ai-gpu-quant-test:py311 \
  bash -lc 'cd /workspace && \
    python -m tests.llm_e2e.run \
      --config tests/llm_e2e/config.gpu.json \
      --deployment reports/qwen-vllm-fp16-server/deployment.json \
      --output reports/qwen-vllm-fp16-test'
```

它测试短、长、混合文本 × 1/2/4/8 并发，记录 TTFT、E2E、TPOT 估计值、输出 token 吞吐、成功率和 GPU CSV。首次没有 SLA 时状态为 `NEEDS_CONFIGURATION`，不是通过。

## 6. Qwen：vLLM BitsAndBytes 4-bit 量化

停止 FP16 服务：

```bash
docker stop qwen-vllm-fp16
```

启动 BitsAndBytes 4-bit。它仍使用同一份原始 FP16/FP32 模型快照，运行时加载为 4-bit；不要把这个结果写成 GPTQ/AWQ 或 TensorRT-LLM 量化结果：

```bash
docker run --rm --gpus all --ipc=host --name qwen-vllm-bnb4 \
  -v "$PWD:/workspace" -v /models:/models -p 8000:8000 \
  ai-gpu-quant-test:py311 \
  bash -lc 'cd /workspace && \
    python -m tests.llm_e2e.serve \
      --model-dir /models/Qwen2.5-0.5B-Instruct \
      --cuda-device 0 --dtype half --quantization bitsandbytes \
      --output reports/qwen-vllm-bnb4-server'
```

等待 `READY` 后运行性能测试：

```bash
mkdir -p reports/qwen-vllm-bnb4-test

docker run --rm --network host \
  -v "$PWD:/workspace" \
  ai-gpu-quant-test:py311 \
  bash -lc 'cd /workspace && \
    python -m tests.llm_e2e.run \
      --config tests/llm_e2e/config.gpu.json \
      --deployment reports/qwen-vllm-bnb4-server/deployment.json \
      --output reports/qwen-vllm-bnb4-test'
```

性能报告可比较 FP16 与 BNB4 的 TTFT/E2E/吞吐，但必须使用相同模型 hash、输入场景、并发矩阵、GPU、驱动和 vLLM 版本。vLLM runtime、量化方式和启动参数属于受控条件，FP16 baseline 不能直接复用原项目 TensorRT 或其他 runtime 的 baseline。

## 7. Qwen：固定任务质量测试

性能通过不代表量化质量通过。先对 FP16 服务捕获固定任务 reference：

```bash
# FP16 服务运行中
docker run --rm --network host \
  -v "$PWD:/workspace" \
  ai-gpu-quant-test:py311 \
  bash -lc 'cd /workspace && \
    python -m tests.model_deployment.qwen_quality \
      --deployment reports/qwen-vllm-fp16-server/deployment.json \
      --url http://127.0.0.1:8000 \
      --output reports/qwen-quality-fp16-reference'
```

结果应为 `REFERENCE_CAPTURED`，然后停止 FP16、启动 BNB4，再比较：

```bash
docker run --rm --network host \
  -v "$PWD:/workspace" \
  ai-gpu-quant-test:py311 \
  bash -lc 'cd /workspace && \
    python -m tests.model_deployment.qwen_quality \
      --deployment reports/qwen-vllm-bnb4-server/deployment.json \
      --url http://127.0.0.1:8000 \
      --reference reports/qwen-quality-fp16-reference/quality.json \
      --max-accuracy-drop-pp 1.0 \
      --output reports/qwen-quality-bnb4'
```

这个质量 smoke 使用 48 个固定算术/信息抽取任务，只能检查服务输出、模型身份和一个极小的可判定任务集，不能替代真实业务评测。正式验收要使用业务 prompts、分类/抽取数据集、格式遵循率、拒答率和人工/模型评测，并把量化质量门禁单独批准。

## 8. 建议质量门禁

| 层级 | DistilBERT | Qwen |
|---|---|---|
| 启动 | 模型 hash、量化模块、readiness inference | 模型 hash、vLLM runtime、readiness generation |
| 正确性 | shape、有限值、FP32 allclose、输入边界 | HTTP/SSE、finish、无串请求、固定任务质量 |
| 量化 | max/mean abs、relative L2、cosine、任务 F1/accuracy | 任务准确率/格式率、输出长度和失败率 |
| 性能 | E2E p50/p95、成功率、峰值显存 | TTFT/E2E/TPOT p50/p95、output tokens/s、GPU 显存 |
| 并发 | 1/2/4/8，响应归属 | 1/2/4/8，流式响应归属和吞吐 |
| 发布 | 没有批准阈值不判 PASS | 没有批准 SLA/质量预算不判 PASS |

初始硬门禁：模型身份必须匹配；服务请求成功率 100%；无 NaN/Inf；非法输入按契约拒绝；请求失败后可恢复；量化模块实际加载；必测 case 完整；报告原始数据齐全。性能和精度阈值必须在 FP32/FP16 基线与业务 SLA 基础上批准，不能使用本 README 示例值直接放行。

## 9. 报告目录

```text
reports/
├── distilbert/fp32-reference/references.json
├── distilbert/fp16/summary.json
├── distilbert/int8/summary.json
├── distilbert/nf4/summary.json
├── qwen-vllm-fp16-server/deployment.json, server.log, gpu.csv
├── qwen-vllm-fp16-test/summary.json, requests.jsonl
├── qwen-vllm-bnb4-server/deployment.json, server.log, gpu.csv
├── qwen-vllm-bnb4-test/summary.json, requests.jsonl
├── qwen-quality-fp16-reference/quality.json
└── qwen-quality-bnb4/quality.json
```

报告里必须保存：模型 revision/hash、tokenizer/config hash、GPU UUID、驱动、CUDA、PyTorch、Transformers、vLLM/bitsandbytes 版本、量化参数、输入数据 hash、并发等级、原始请求和失败原因。

## 10. 常见问题

- **4090D OOM**：先停止另一个服务；降低 `--max-model-len`、`--max-num-seqs` 或 `--gpu-memory-utilization`；不要用 OOM 重试结果当性能数据。
- **bitsandbytes 没有 CUDA**：检查容器中的 `torch.cuda.is_available()`、`bitsandbytes` 版本和 `nvidia-smi`；不要把 CPU fallback 当 GPU 量化测试。
- **vLLM 不支持某参数**：查看 `vllm serve --help=all`，启动器会写 `cli-help.txt` 并停止；按实际版本更新适配，不要删除检查后强行启动。
- **DistilBERT INT8/NF4 误差大**：先看层级/输出误差和任务指标；不要直接把 `atol/rtol` 调大到通过。
- **Qwen BNB4 文本不同**：固定 temperature=0 仍可能有合法数值差异；用固定任务质量和分布/格式指标，不能只比较完整字符串。
- **TTFT 很高**：区分客户端测到的排队、tokenizer、网络和 prefill；GPU 利用率低不代表服务没有排队。
- **并发结果失败**：先检查服务端 OOM、max-num-seqs、请求 timeout 和 GPU CSV；再判断是调度/资源问题还是模型输出问题。
