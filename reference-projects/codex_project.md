# PyTorch GPU 算子验证与性能回归质量门禁

## 项目定位

这是一个面向 Deep Learning Software Test、PyTorch/CUDA QA 和 GPU 测试开发岗位的自动化验证平台。平台以 PyTorch CPU/reference 实现和 GPU 实现为对照，在固定测试矩阵下验证算子正确性、数值精度、梯度、边界行为、显存和性能回归，并生成可复现的测试结论。

V1 只覆盖 MatMul、LayerNorm、Softmax、RMSNorm 四类算子，不宣称覆盖全部 PyTorch 算子。

## 解决的痛点

- CUDA、PyTorch 或算子实现升级后结果错误不易发现。
- FP16/BF16 与 FP32 的误差不能用精确相等判断。
- 特殊 shape、非连续 Tensor、空输入和极端数值经常被遗漏。
- GPU 异步执行导致性能计时不准确。
- 性能退化只能人工查看日志，缺少版本化 baseline 和回归判定。

## 受测对象与测试矩阵

每个算子至少覆盖：

| 维度 | 示例 |
| --- | --- |
| shape | 小、中、大矩阵；非对齐尺寸；batch=1 和大 batch |
| dtype | FP32、FP16；GPU 支持时增加 BF16 |
| device | CPU reference、CUDA candidate |
| layout | contiguous、transpose 后的 non-contiguous Tensor |
| 输入 | 随机值、全零、极端值、NaN/Inf（按算子语义处理） |

测试固定随机种子，并保存 case id、shape、dtype 和输入摘要，失败时可以复现。

## 技术栈与环境

- Python 3.10+
- PyTorch 2.x
- Pytest
- NumPy、pandas
- CUDA 11.8/12.x（以云 GPU 实际环境锁定）
- pynvml 或 NVML
- torch.profiler
- Docker

本地执行 CPU 单元测试；云 NVIDIA GPU 执行 CUDA、FP16/BF16 和性能测试。报告记录 GPU 型号、驱动、CUDA、PyTorch、git commit 和容器 image digest。不同硬件或软件 fingerprint 的性能只标记 `NOT_COMPARABLE`，不直接比较。

## 仓库交付物

```text
pytorch-gpu-validation-gate/
  src/
    operators.py              # 统一 reference/candidate 接口
    case_generator.py         # shape/dtype/layout 测试矩阵
    benchmark.py              # warmup、CUDA Event、统计
    gpu_monitor.py            # 显存、利用率和设备信息
  tests/
    test_correctness.py
    test_precision.py
    test_gradients.py
    test_boundary.py
    test_performance.py
  baselines/<gpu>/<env>.json
  reports/<run-id>/summary.json
  scripts/run_suite.py
  Dockerfile
  requirements.txt
  README.md
```

## 执行链路

```text
生成固定测试 case
  ↓
运行 CPU/PyTorch reference
  ↓
运行 CUDA candidate
  ↓
检查 shape、dtype、NaN/Inf 和数值误差
  ↓
检查梯度与边界行为
  ↓
warmup 后使用 CUDA Event 测量性能
  ↓
采集显存/GPU 指标
  ↓
与同一 fingerprint 的 baseline 比较
  ↓
生成 PASS、BLOCKED 或 NOT_COMPARABLE 报告
```

## 测试与判定规则

### 正确性与精度

使用 PyTorch reference 对比 candidate，记录 `max_abs_error`、`mean_abs_error`、`relative_l2_error`。FP32/FP16 使用经过首轮实验确认并版本化的 `rtol/atol`，不为让测试通过临时放宽阈值。

### 梯度

对可训练算子比较前向输出和输入/参数梯度。测试前清空梯度，测试后验证没有 NaN/Inf。

### 边界与异常

覆盖 batch=1、非对齐尺寸、零维或最小合法输入、non-contiguous Tensor、错误 dtype、非法 shape 和预期 OOM。每类异常断言清晰的失败类型，而不是只断言“程序报错”。

### 性能

每个 case 先 warmup，再在同一 CUDA Stream 使用 `torch.cuda.Event` 测量多次原始样本，统计 median、p95、mean、标准差和吞吐。性能回归只在 GPU、驱动、CUDA、PyTorch 和容器 fingerprint 一致时启用。MVP 先标记 `PERFORMANCE_REVIEW_REQUIRED`，V1 再基于历史 trial 和置信区间升级为阻断规则。

### 报告

报告至少包含：git commit、环境 fingerprint、case、输入摘要、正确性指标、性能原始样本、显存峰值、失败原因和复现命令。测试报告同时生成 JUnit XML，便于 CI 展示。

## CI 与复现

- Pull Request：CPU 单元测试和少量 CUDA smoke。
- Nightly：完整 correctness、precision、performance 和稳定性套件。
- GPU 性能 gate 只在 fingerprint 与批准 baseline 一致时生效。

```bash
pytest -m "not gpu"
pytest tests --gpu
python scripts/run_suite.py --suite correctness,performance
```

## 分阶段交付

### MVP

完成 MatMul/LayerNorm 的 correctness、precision、boundary 和 CUDA Event benchmark，生成 JSON/Markdown 报告，并在一张云 GPU 上真实运行。

### V1

增加 Softmax/RMSNorm、梯度测试、NVML 监控、性能 baseline、Docker 和一次故意制造的性能回归案例。

## 面试展示

展示一次 FP16 容差判断、一次 device/layout 边界失败，以及一次人为引入同步导致的性能退化。重点解释测试矩阵、CUDA 计时、误差阈值和 baseline 可比性。

## 简历描述

开发 PyTorch GPU 算子验证与性能回归质量门禁，基于 Pytest 构建多 shape、多 dtype、多 layout 测试矩阵，覆盖前向、梯度、边界、数值误差和 CUDA Event 性能测试；集成 NVML、Docker 和版本化 baseline，自动输出 PASS/BLOCKED/NOT_COMPARABLE 结果。



# Triton/TensorRT GPU 算子与推理发布质量门禁

## 项目定位

这是一个面向 NVIDIA TensorRT QA、GPU/CUDA 测试、推理性能和 AI 基础设施测试岗位的端到端项目。项目分为 Triton kernel 专项和 TensorRT 推理质量门禁两层：先验证自定义 GPU kernel 的正确性与性能，再验证同一模型转换为 TensorRT FP16 Engine 后是否具备正确、兼容、稳定和可发布的质量。

第一阶段不加入 INT8 校准、多机多卡、SSE 或复杂 TensorRT-LLM 协议；这些属于后续扩展，避免没有真实实现时扩大承诺。

## 解决的痛点

- 自定义 kernel 可能比 reference 快，但结果不准确。
- 某些 shape 或 dtype 下 kernel 性能突然退化。
- PyTorch → ONNX → TensorRT 转换后可能出现输出误差或 shape 不兼容。
- Engine 只验证“能跑”，没有验证 profile 外输入和失败行为。
- 性能数据受 GPU、驱动和 TensorRT 版本影响，跨环境比较会产生误导。
- 长时间运行可能出现资源增长、CUDA error 或输出漂移。

## 受测对象

### Triton

V1 实现 Vector Add 与 Softmax 或 LayerNorm，支持 FP32/FP16 和固定范围的 shape。每个 kernel 都有 PyTorch reference、随机 correctness、误差、边界和 benchmark 测试。

### TensorRT

使用自维护的 TinyTransformerEncoder 或小型 BERT/MLP 模型，包含 Embedding、MatMul、Softmax、LayerNorm、GELU 和线性层。固定随机种子、`eval()` 模式、权重 hash 和输入 case。

输入 profile 明确写入报告，例如 batch 1-8、sequence 8-512，使用 min/opt/max optimization profile。项目只宣称覆盖已验证的模型和 profile。

## 技术栈与环境

- Python 3.10+
- PyTorch 2.x
- Triton
- ONNX
- TensorRT 10.x（按实际云 GPU 环境锁定）
- CUDA 12.x 或实际可用版本
- Nsight Systems；可选 Nsight Compute
- Docker、Pytest、NVML

真实执行必须使用 NVIDIA GPU。报告记录 GPU 型号、driver、CUDA、TensorRT、PyTorch、容器 image digest、模型 SHA-256、Engine SHA-256 和 git commit。

## 仓库交付物

```text
triton-tensorrt-inference-gate/
  triton_kernels/
    vector_add.py
    softmax.py
  src/
    model.py
    export_onnx.py
    build_engine.py
    trt_runner.py
  tests/
    test_triton_correctness.py
    test_triton_performance.py
    test_trt_correctness.py
    test_dynamic_shapes.py
    test_negative_cases.py
    test_performance.py
    test_soak.py
  baselines/<gpu>/<environment>.json
  scripts/run_suite.py
  reports/<run-id>/summary.json
  reports/<run-id>/release-summary.md
  docker/Dockerfile
  README.md
```

## 真实执行链路

### Triton 层

```text
生成输入
  ↓
运行 PyTorch reference
  ↓
运行 Triton kernel
  ↓
比较误差、NaN/Inf 和 shape
  ↓
CUDA Event benchmark
  ↓
Nsight 分析 block size、带宽和 kernel 时间
```

### TensorRT 层

```text
固定模型和输入 case
  ↓
导出 ONNX 并运行 checker
  ↓
构建 FP16 Engine 和 optimization profile
  ↓
设置实际 input shape 和 tensor address
  ↓
execute_async_v3(stream)
  ↓
与 PyTorch FP32 reference 对比
  ↓
动态 shape、异常、性能和稳定性测试
  ↓
生成发布结论
```

TensorRT 执行期间必须保持输入/输出 Tensor 的 Python 引用，直到对应 CUDA stream 完成；“复用 PyTorch CUDA 地址”不等于绕过 GPU 内存访问。

## 测试与判定规则

### Triton correctness

覆盖随机输入、全零、极端值、非对齐 shape、FP32/FP16。记录 `max_abs_error`、`mean_abs_error`、relative L2 和失败 case id。正确性未通过时禁止讨论性能提升。

### TensorRT correctness

输出 shape/dtype 必须符合契约，禁止 NaN/Inf。使用版本化的 FP16 容差，不把 FP64 CPU 结果不加说明地称为绝对金标准。

### Dynamic shape 与负向测试

正向覆盖 min/opt/max 和 profile 内非对齐 shape。负向覆盖 profile 外 shape、batch=0、错误 dtype、缺少 tensor name 和损坏 Engine，并断言预期异常类型。

### 性能

每种 shape 先 warmup，再用同一 CUDA Stream 的 CUDA Event 采集多次设备端 latency，统计 median、p95、mean、标准差和吞吐。只在 GPU/driver/CUDA/TensorRT/image fingerprint 一致时比较 baseline。首次运行只生成候选 baseline，人工审阅后批准；性能信号可以先为 `PERFORMANCE_REVIEW_REQUIRED`，不要直接用未经验证的绝对阈值阻断发布。

### 稳定性

在 opt shape 连续执行 1000 次，检查 CUDA error、输出漂移、NaN/Inf、free memory、max allocated memory 和 RSS。没有 profiler 证据时称为“资源增长检测”，不武断宣称发现 C++ 内存泄漏。

## 发布状态

```text
PASS              所有适用测试通过
BLOCKED           发现正确性、契约、稳定性或已验证的性能回归
NOT_COMPARABLE   环境 fingerprint 与 baseline 不同，不能作性能结论
```

`release-summary.md` 面向发布负责人，必须包含总体状态、失败 case、受影响 shape、环境差异、原始报告路径和复现命令。`summary.json` 保存完整证据。

## CI 与复现

- Pull Request：CPU/ONNX checker、Triton smoke 和少量 GPU smoke。
- Nightly：完整 correctness、dynamic、performance 和 soak。
- GPU runner 使用 `self-hosted`、`gpu` 标签。
- 环境不一致时只输出 `NOT_COMPARABLE`，不输出误导性性能结论。

```bash
docker build -t trt-regression:local -f docker/Dockerfile .
docker run --gpus all --rm -v "$PWD:/workspace" trt-regression:local \
  python scripts/run_suite.py --suite triton,trt,dynamic,performance,soak
```

## 分阶段交付

### 第 1 阶段：Triton MVP

完成 Vector Add 和 Softmax/LayerNorm 的 correctness、FP16 误差和 benchmark。

### 第 2 阶段：TensorRT MVP

完成 TinyTransformerEncoder/小型 BERT 的 ONNX 导出、FP16 Engine、PyTorch 对比和动态 shape 测试。

### 第 3 阶段：作品集 V1

加入负向测试、1000 次 soak、NVML、Docker、报告、baseline 和一次故意制造的性能回归。时间允许时再加入 Qwen2.5-1.5B 的 vLLM 或 TensorRT-LLM 推理 smoke。

## 面试展示

1. 展示 Triton kernel 与 PyTorch reference 的误差和 shape 覆盖。
2. 解释 CUDA Event、warmup、stream 和 GPU 异步执行。
3. 演示 profile 外 shape 被正确拒绝。
4. 展示一次人为加入同步或重复执行导致的性能退化。
5. 解释为什么不同 GPU/TensorRT 版本的结果是 `NOT_COMPARABLE`。
6. 说明 tensor address 生命周期和 `execute_async_v3` 的同步边界。

## 简历描述

开发 Triton/TensorRT GPU 推理发布质量门禁，覆盖自定义 kernel correctness、FP16 数值误差、ONNX→TensorRT Engine 构建、动态 Shape、负向输入、1000 次稳定性和 CUDA Event 性能回归；通过 Docker、环境 fingerprint、模型/Engine hash 和 PASS/BLOCKED/NOT_COMPARABLE 报告实现可复现的 GPU 推理质量验证。
