# GPU 推理发布质量门禁：可复现的 TensorRT 正确性、兼容性与性能回归平台

## 项目定位

这是一个面向 NVIDIA GPU 推理软件质量岗位的测试开发项目。目标不是展示概念，而是交付一个任何具备 NVIDIA GPU 的人都能复跑、得到同类结论的推理发布质量门禁。

平台分三层验证同一组推理算子（`MatMul`、`LayerNorm`、`Softmax`、`GELU`）：先在算子级别对比 PyTorch CPU/GPU 参考实现，再验证其中一个算子的自定义 Triton kernel 实现，最后把整个模型编译为 TensorRT FP16 Engine 做端到端发布验证。三层共享同一套判定语言（`PASS`/`BLOCKED`/`NOT COMPARABLE`），确保结论互相印证，而不是三份互不相关的报告。

- 算子级正确性与梯度矩阵：对固定算子集合按 shape×dtype×layout×device 组合验证前向数值与反向梯度。
- 自定义 Triton kernel 验证：至少一个算子的 Triton 实现对比 PyTorch reference 的正确性与性能。
- 数值正确性对标：PyTorch FP32 参考结果与 TensorRT FP16 结果的差异受控。
- 动态 Shape 与异常输入兼容性验证：合法 Profile 内的输入通过，Profile 外输入得到预期失败。
- GPU 设备端性能回归：以 CUDA Event 测量真实推理，按版本基线阻断显著退化。
- 长循环稳定性验证：监测可用显存、错误与输出稳定性，发现资源持续增长或结果漂移。
- 可审计报告：每一次结果都带有代码提交、模型哈希、GPU、驱动、CUDA、TensorRT 和测试参数。

项目名称建议写作：**TensorRT Inference Release Quality Gate**。

## 用户、决策与项目价值

### 核心用户

- **推理工程师：**修改 ONNX 图、TensorRT Builder 配置、CUDA/TensorRT 版本后，需要确认 Engine 是否仍可安全交付。
- **QA 或发布负责人：**需要基于统一规则决定候选 Engine 能否进入预发布环境，而不是人工翻看零散日志。
- **Runtime 性能工程师：**需要在出现性能退化时，拿到受影响 workload、环境差异和可复制的复现命令。

### 核心任务（Job to Be Done）

当模型、Engine、CUDA、驱动或 TensorRT 发生变化时，发布负责人应能在一次标准运行后回答：**这个候选版本能否发布；若不能，问题是正确性、输入契约、稳定性还是性能；任何工程师如何在同一环境中复现。**

平台将原本依赖人工经验的验证过程转化为有依据的发布结论：`PASS`、`BLOCKED` 或 `NOT COMPARABLE`。它的业务价值不是孤立追求更低延迟，而是减少问题 Engine 进入下一环境的概率，并缩短定位与复验时间。

## 为什么这个范围适合测试开发岗位

项目覆盖的是推理 QA 和 GPU 软件测试中真实且可面试追问的闭环：算子级数值与梯度验证、自定义 kernel 正确性、参考实现设计、覆盖策略、环境兼容性、性能测量、异常注入、回归判定、证据留存、发布决策和 CI 执行。

算子矩阵仅覆盖模型中实际使用的四类算子（`MatMul`、`LayerNorm`、`Softmax`、`GELU`），Triton kernel 仅覆盖其中一个算子的自定义实现，不宣称覆盖 PyTorch 全部算子或通用 kernel 库。刻意不在第一阶段加入 INT8、SSE 和大模型多机压测。它们涉及校准质量、具体服务端后端与协议语义；没有真实实现时，只会让测试边界不清晰。三层闭环做扎实后，再扩展更有说服力。

## 受测对象与固定环境

### 模型

使用一个自维护的 `TinyTransformerEncoder`，包含 Embedding、MatMul、Softmax、LayerNorm、GELU 和线性层。模型固定随机种子、`eval()` 模式和权重文件哈希，避免 Dropout 或权重变化污染回归结果。

首个已验证 workload 仅是 `TinyTransformerEncoder`，不能宣称覆盖所有推理模型。V1 使用同一执行与报告接口增加 `ResNet-18` 或简化 ONNX MLP 作为第二个 workload，用于验证框架可适配不同模型图，而不是只适配一个 Engine。

第一阶段只支持一套明确的输入契约：

| 输入 | dtype | Shape profile |
| --- | --- | --- |
| `input_ids` | `int32` | batch: 1-8, sequence: 8-512 |
| `attention_mask` | `int32` | 与 `input_ids` 相同 |
| `hidden_states` 输出 | `float16` | batch: 1-8, sequence: 8-512, hidden: 256 |

`min=(1, 8)`、`opt=(4, 128)`、`max=(8, 512)` 是 Engine 构建时传入的 optimization profile，不把“任意动态 Shape”写成不可证明的承诺。

### 算子测试矩阵

模型中实际使用的四类算子（`MatMul`、`LayerNorm`、`Softmax`、`GELU`）在算子级别单独验证，覆盖：

| 维度 | 取值 |
| --- | --- |
| shape | 小（128×128）、中（1024×1024）、大（4096×4096）；batch=1 与 batch=8；非对齐尺寸（如 1023） |
| dtype | FP32、FP16 |
| layout | contiguous；transpose 后的 non-contiguous Tensor |
| device | CPU reference、CUDA candidate |
| 输入 | 随机正态分布、全零、极端值（如 1e4、-1e4）、按算子语义处理的 NaN/Inf 探测 |

每个算子的每种组合都固定随机种子，并保存 case id、shape、dtype、layout 供失败复现。梯度测试只覆盖可训练算子（`MatMul`、`LayerNorm`、`GELU`）：对比前向输出以及输入、参数的梯度，测试前清空历史梯度，测试后断言梯度中没有 NaN/Inf。

这一层的参考实现是 PyTorch CPU FP32；不覆盖 BF16、INT8 或本项目模型未使用的算子，避免把范围扩大到无法验证的承诺。

### Triton 自定义 Kernel

为证明具备自定义 GPU kernel 的开发与验证能力，使用 Triton 语言重新实现 `Softmax`（沿最后一维归约），并将其作为 TensorRT Engine 之外的独立可选执行路径。

验证内容：

- **正确性：**对比 PyTorch `softmax` reference，覆盖上表中的 shape 与 FP32/FP16，记录 `max_abs_error` 和 `relative_l2_error`。
- **边界：**验证极端输入（全零、大数值差、单元素行）不产生 NaN/Inf 或除零错误。
- **性能：**用 `torch.cuda.Event` 对比 Triton kernel 与 `torch.nn.functional.softmax` 的设备端延迟，只做同 GPU、同 dtype 下的相对对比，不代表通用 kernel 性能结论。

V1 只实现一个 Triton kernel；不实现多 kernel 融合、autotune 网格搜索或跨架构适配，避免自定义 kernel 部分与算子矩阵和 TensorRT Engine 部分抢时间。

### 可复现环境

以 Docker 镜像作为唯一运行环境，镜像标签、`nvidia-smi`、CUDA、cuDNN、TensorRT、PyTorch、ONNX 和 Python 版本写入每份报告。基线只在同一 GPU 型号、相同 power limit、相同镜像标签下比较。

建议首个已验证环境是：Ubuntu 22.04、CUDA 12.x、TensorRT 10.x、PyTorch 2.x 与一张实际可用的 NVIDIA GPU。版本号在首次成功运行后锁定到 `docker/Dockerfile` 和报告中，不能编造型号或版本。

## 仓库交付物

```text
ai-gpu-test-lab/
  docker/Dockerfile
  requirements.txt
  src/operators.py                     # MatMul/LayerNorm/Softmax/GELU 的 reference/candidate 接口
  src/case_generator.py                # shape x dtype x layout x device 矩阵生成
  src/model.py                         # 固定的 TinyTransformerEncoder
  src/export_onnx.py                   # 生成 model.onnx 与模型哈希
  src/build_engine.py                  # 生成 FP16 model.plan 和 profile 元数据
  src/trt_runner.py                    # TensorRT IRuntime/IExecutionContext 执行器
  triton_kernels/softmax.py            # 自定义 Triton Softmax kernel
  tests/test_operator_matrix.py         # shape x dtype x layout x device 正确性
  tests/test_operator_gradients.py      # 可训练算子的梯度验证
  tests/test_triton_kernel.py           # Triton kernel 正确性与性能
  tests/test_correctness.py             # 数值、NaN/Inf、确定性
  tests/test_dynamic_shapes.py          # min/opt/max 与 profile 外拒绝
  tests/test_negative_cases.py          # dtype、名称、shape、损坏 engine
  tests/test_performance.py             # CUDA Event 性能回归
  tests/test_soak.py                    # 1000 次稳定性循环
  testdata/cases.npz                   # 固定种子生成并提交哈希的输入集
  baselines/<gpu>/<image-tag>.json      # 已批准的性能基线
  scripts/run_suite.py                  # 统一入口与报告落盘
  reports/                              # CI 产物，不提交到 Git
  .github/workflows/gpu-regression.yml  # self-hosted GPU runner
  README.md                             # 一条命令、限制和结果解释
```

全部 `.plan`、ONNX 和报告都由脚本生成；仓库只提交生成脚本、小型固定测试数据及其 SHA-256，不提交来源不明的大型模型或虚构结果。

## 真实执行链路

1. `export_onnx.py` 在固定种子、`eval()`、opset 固定的条件下导出 ONNX，并运行 ONNX checker。
2. `build_engine.py` 创建 TensorRT Builder Config，开启 `FP16`，设置上表中的 optimization profile，序列化 `model.plan`，记录 builder 配置和 Engine SHA-256。
3. `trt_runner.py` 反序列化 Engine，创建 Execution Context 和一个显式 CUDA Stream。
4. 每次调用根据实际输入 Shape 设置 input shape，将 PyTorch CUDA tensor 的 `data_ptr()` 绑定至 TensorRT named tensor address；输出 tensor 由 PyTorch 在同一设备预分配。
5. 调用 `execute_async_v3(stream_handle)`，随后只在需要读取结果或计时窗口结束时同步该 stream。

这里的“零拷贝”只指测试运行时复用 PyTorch 已分配的 CUDA tensor 地址，不等同于绕过所有设备内存访问。代码必须明确保存 input/output tensor 的 Python 引用直到 TensorRT stream 完成，避免地址生命周期错误。

## 测试设计与判定规则

### 1. 算子级测试矩阵判定规则

对每个算子、每个 shape×dtype×layout×device 组合：

- 输出 shape、dtype 与 PyTorch reference 一致，且不存在 NaN/Inf（除非算子语义允许，如极端输入下的溢出需显式断言，而不是被忽略）。
- FP32 使用 `atol=1e-6`、`rtol=1e-5`；FP16 使用经首轮实验校准并版本化的容差（初始建议 `atol=1e-3`、`rtol=1e-3`），不为通过测试放宽。
- non-contiguous layout 下结果必须与 contiguous 一致，用来暴露只在特定内存布局下出现的实现错误。
- 梯度测试对比 `.grad` 与 PyTorch reference 的 `.grad`，误差阈值与前向一致；梯度中出现 NaN/Inf 直接判定失败。

任一组合失败都记录算子名、shape、dtype、layout、device 和误差值，便于复现，不允许只报告“部分用例失败”而不列出具体组合。

### 2. Triton Kernel 判定规则

Triton Softmax kernel 必须先通过与算子矩阵相同的正确性和边界断言，才允许报告性能数据。性能对比只在同一 GPU、同一 dtype 下进行，使用与 TensorRT 性能测试相同的 warmup、CUDA Event 和多次采样方法。若 Triton kernel 比 PyTorch 实现慢，如实记录，不能只展示更快的场景。

### 3. TensorRT Engine 正确性测试

参考实现是同一权重的 PyTorch FP32 GPU `eval()` 推理；输入按相同语义转换到 FP32。第一阶段不把 FP64 CPU 称为绝对金标准，因为实际 GPU 算子、累加顺序和 TensorRT 融合均会改变浮点舍入路径。

测试样本包含固定的正常 token、全零 mask、最大 token id、短序列、最大序列和固定种子随机样本。对每个 shape 执行：

- 输出 shape 和 dtype 符合契约。
- 输出中不存在 NaN 或 Inf。
- 计算 `max_abs_error`、`mean_abs_error`、`relative_l2_error` 与余弦相似度。
- 使用已验证的 FP16 容差，例如 `torch.testing.assert_close(actual, reference, rtol=5e-3, atol=5e-3)`；容差通过首轮实验数据确定后版本化，不为让测试通过而临时放宽。

报告既记录通过/失败，也记录每个指标。阈值失败时保存输入 case id、shape、Engine hash 和最大误差位置，供定位复现。

### 4. Dynamic Shape 和负向测试

对 `(1, 8)`、`(4, 128)`、`(8, 512)` 以及 Profile 内的非对齐 shape（如 `(3, 127)`）进行正向测试。对 batch=0、sequence=513、错误 dtype、缺少 tensor name、损坏 Engine 文件分别断言预期异常类型和清晰错误信息。

这组测试证明平台不仅验证“能跑”，也验证输入契约与失败行为。

### 5. 性能测试

性能测量只包围真正的 `execute_async_v3` 调用。每种 shape：

1. 先运行 100 次 warmup，不计入结果。
2. 使用同一 CUDA Stream 上的 `torch.cuda.Event(enable_timing=True)` 测量 500 次真实推理。
3. 记录单次设备端延迟的 median、p95、mean、standard deviation 与吞吐量。
4. 将原始样本写入 JSON，方便复算统计值。

性能基线不使用“固定小于 3 ms”这种跨 GPU 无意义的绝对断言。对于同一个 GPU 和镜像标签，连续运行 5 个独立 trial，将各 trial 的 median 组成样本。候选版本同时满足以下条件时，标记为 `PERFORMANCE REVIEW REQUIRED`：

$$
\frac{\operatorname{median}(L_{candidate})}
     {\operatorname{median}(L_{baseline})} > 1.08
$$

且 candidate/baseline 的 bootstrap 95% 延迟区间不重叠。MVP 不因该信号自动阻断发布，而是要求人工复验；首次运行只生成候选基线，需人工审阅后才能批准为 gate。

V1 仅在独占 GPU、锁定 power/clock、积累足够历史 trial 后，才将经过验证的性能规则升级为 `BLOCKED`。基线升级必须保留原因、提交号与审批记录。报告始终区分设备端 kernel latency 与端到端请求 latency，避免将其中一个误解为另一个。

### 6. 稳定性与资源测试

在 `opt=(4, 128)` 下连续执行 1000 次真实推理，周期性同步 stream，检查输出没有 NaN/Inf。循环前后记录 `torch.cuda.mem_get_info()` 的 free/total memory、`torch.cuda.max_memory_allocated()` 和进程 RSS。

判定标准是：无 CUDA error、无输出漂移，并且清理 Engine/Context 后设备可用显存相对初始值的差异不超过预先定义的容差。这个测试称为“资源增长检测”，而不是在没有 profiler 证据时武断宣称发现 C++ 显存泄漏。

## 报告与证据格式

每次 `scripts/run_suite.py` 生成 `reports/<run-id>/summary.json`、JUnit XML 和面向发布负责人的 `release-summary.md`。`summary.json` 至少包含：

```json
{
  "git_commit": "<real commit>",
  "model_sha256": "<hash>",
  "engine_sha256": "<hash>",
  "environment": {
    "gpu_name": "<nvidia-smi result>",
    "driver": "<version>",
    "cuda": "<version>",
    "tensorrt": "<version>",
    "container_image": "<digest>"
  },
  "case": {"batch": 4, "sequence": 128},
  "correctness": {"max_abs_error": 0.0, "relative_l2_error": 0.0},
  "performance_ms": {"median": 0.0, "p95": 0.0},
  "status": "pass"
}
```

JSON 中的 `0.0` 是字段示例，不能作为项目结果。README 放一张由真实 CI 产物导出的对比表，并链接或附带可下载的原始报告。

`release-summary.md` 只展示发布决策所需信息：总体状态、与批准基线相比的正确性/性能变化、受影响的 shape、失败 case、环境差异、原始报告路径和一条复现命令。例如：

```text
Release decision: BLOCKED
Reason: input shape (8, 513) exceeds the approved optimization profile
Correctness: PASS for all in-profile cases; max_abs_error=0.0031
Performance: NOT COMPARABLE (TensorRT version differs from baseline)
Reproduce: docker run ... python scripts/run_suite.py --case batch=8,sequence=513
```

发布负责人不需要阅读原始 JSON 才能做决定；工程师则可从摘要跳转到对应原始样本与环境信息完成定位。

## CI 与执行方式

GPU 测试在贴有 `self-hosted` 和 `gpu` 标签的 Runner 上运行。Pull Request 默认执行 ONNX 合规、CPU 单元测试和少量 GPU smoke，并把 `release-summary.md` 作为 PR 检查结果；每日定时任务执行完整正确性、性能和 1000 次稳定性测试。性能 gate 只在环境 fingerprint 与批准基线完全一致时启用，否则标记为 `NOT COMPARABLE`，不输出误导性结论。

建议的验收命令为：

```bash
docker build -t trt-regression:local -f docker/Dockerfile .
docker run --gpus all --rm -v "$PWD:/workspace" trt-regression:local \
  python scripts/run_suite.py --suite correctness,dynamic,performance,soak
```

## 14 天详细实施计划

### MVP 与 V1 边界

**MVP 完成条件（即本 14 天计划的产出）**：算子级测试矩阵（四类算子 × FP32/FP16 × contiguous/non-contiguous × CPU/CUDA）全部跑通并生成误差报告；一个 Triton kernel 通过正确性与性能验证；本地 Docker 环境中真实构建并执行一个 TensorRT FP16 Engine，产出正确性、动态 Shape、负向与稳定性测试结果；一份 `release-summary.md`；一次可运行的 CI；一个故意制造且可复现的失败案例。它不要求长期可用的 GPU 集群，只要求 14 天内至少能稳定访问一张 NVIDIA GPU（本地或云实例均可）。

**V1（14 天之后再做）**：增加第二个 workload（如 ResNet-18 或 ONNX MLP）、把性能信号从 `PERFORMANCE REVIEW REQUIRED` 升级为经过历史数据验证的 `BLOCKED` 阻断 gate、以及可选的 Triton 服务契约扩展。

### 全局前置知识清单

开工前先确认自己能用一两句话解释以下概念，不理解的先看对应文档再动手，避免边查边写导致进度失控：

| 主题 | 需要掌握的点 | 参考 |
| --- | --- | --- |
| PyTorch CUDA 基础 | `tensor.cuda()`、`.contiguous()`/`.T` 产生的 non-contiguous 视图、`torch.cuda.synchronize()`、`autograd`、`.grad` | PyTorch 官方文档 Tensor/Autograd 章节 |
| 数值精度 | FP32/FP16 表示范围与舍入误差、为什么不能用 `==`、`rtol`/`atol` 含义 | `torch.testing.assert_close` 文档 |
| CUDA 计时 | 为什么 Python `time.time()` 测不准异步 GPU 调用、`torch.cuda.Event(enable_timing=True)`、warmup 的作用 | PyTorch Profiler / CUDA Event 文档 |
| ONNX | opset 版本、`dynamic_axes`、`onnx.checker.check_model` | ONNX 官方文档 |
| TensorRT 10.x Python API | `Builder`/`INetworkDefinition`/`OnnxParser`/`IBuilderConfig`/`IOptimizationProfile`/`IExecutionContext`、`set_input_shape`、`set_tensor_address`、`execute_async_v3` | NVIDIA TensorRT Developer Guide |
| CUDA Stream 与生命周期 | 为什么异步调用后要保留 tensor 引用到 stream 完成、`stream.synchronize()` | CUDA Programming Guide（Streams 章节） |
| Triton 语言 | `@triton.jit`、`tl.program_id`、`tl.load`/`tl.store`、`mask`、`BLOCK_SIZE: tl.constexpr` | Triton 官方 tutorials（Softmax 示例） |
| pytest | `fixture`、`parametrize`、`-m` marker、`--gpu` 自定义参数 | pytest 官方文档 |
| Docker + NVIDIA Container Toolkit | `--gpus all`、镜像分层、`nvidia-smi` 在容器内验证 | NVIDIA Container Toolkit 文档 |
| 统计方法 | median/p95、bootstrap 置信区间的直觉理解（不需要推导，只需要知道用来判断两组延迟样本是否有统计学差异） | 任意统计学 bootstrap 简介 |
| CI/CD | GitHub Actions workflow 语法、`self-hosted`、`runs-on` 标签 | GitHub Actions 文档 |

---

### 第 1 天：环境、仓库骨架与算子接口设计

**今日目标**：把仓库骨架建起来，能在 CPU 上跑通一个空的 pytest 套件；设计好算子测试要用的统一接口，这是后面所有测试矩阵的地基。

**验收标准**：
- `pytest tests/ -m "not gpu"` 能运行并全部通过（哪怕大部分是占位测试）。
- `docker build` 能成功构建镜像（即使只装了基础依赖）。
- `src/operators.py` 中每个算子都有 reference（PyTorch CPU）和 candidate（PyTorch CUDA）两个可调用对象，接口签名一致。

**代码骨架**：

```python
# src/operators.py
import torch
import torch.nn.functional as F

class OperatorUnderTest:
    """统一 reference(CPU) / candidate(CUDA) 调用接口，供测试矩阵复用。"""

    name: str

    def reference(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def candidate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class MatMulOp(OperatorUnderTest):
    name = "matmul"

    def __init__(self, k: int, n: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(k, n, generator=g)

    def reference(self, x, **kwargs):
        return x.cpu().float() @ self.weight.float()

    def candidate(self, x, dtype=torch.float16, **kwargs):
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
        w = self.weight.to(device="cuda", dtype=dtype)
        b = self.bias.to(device="cuda", dtype=dtype)
        return F.layer_norm(x.to(device="cuda", dtype=dtype), (x.shape[-1],), w, b)


class SoftmaxOp(OperatorUnderTest):
    name = "softmax"

    def reference(self, x, **kwargs):
        return F.softmax(x.cpu().float(), dim=-1)

    def candidate(self, x, dtype=torch.float16, **kwargs):
        return F.softmax(x.to(device="cuda", dtype=dtype), dim=-1)


class GELUOp(OperatorUnderTest):
    name = "gelu"

    def reference(self, x, **kwargs):
        return F.gelu(x.cpu().float())

    def candidate(self, x, dtype=torch.float16, **kwargs):
        return F.gelu(x.to(device="cuda", dtype=dtype))


OPERATORS = {
    "matmul": lambda: MatMulOp(k=1024, n=1024),
    "layernorm": lambda: LayerNormOp(hidden=1024),
    "softmax": lambda: SoftmaxOp(),
    "gelu": lambda: GELUOp(),
}
```

```text
# docker/Dockerfile（第 1 天先跑通 CPU 部分，第 7 天补 TensorRT/Triton 层）
FROM nvcr.io/nvidia/pytorch:24.08-py3
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
```

**今日测试要点**：只写占位测试确认框架能跑，比如 `test_operator_matrix.py` 里对 `OPERATORS` 字典做一次 `assert callable`。真正的数值断言从第 2 天开始。

---

### 第 2 天：算子级测试矩阵（shape × dtype × layout × device）

**前置知识点**：`.contiguous()`、`.T`/`.transpose()` 产生的 non-contiguous tensor、`torch.testing.assert_close` 的 `rtol`/`atol` 语义。

**今日目标**：实现 `case_generator.py`，为四类算子生成完整测试矩阵；实现 `test_operator_matrix.py`，跑通全部组合的前向正确性。

**验收标准**：
- 4 个算子 × {shape: 小/中/大/非对齐} × {dtype: FP32/FP16} × {layout: contiguous/non-contiguous} × {device: CPU/CUDA candidate} 全部生成用例并执行。
- 每个失败用例的报告里能看到算子名、shape、dtype、layout、`max_abs_error`。
- FP32 用 `atol=1e-6, rtol=1e-5`；FP16 用 `atol=1e-3, rtol=1e-3`（先用这个初始值，后续用真实数据校准）。

**代码骨架**：

```python
# src/case_generator.py
import itertools
from dataclasses import dataclass

import torch

SHAPES = [(128, 128), (1024, 1024), (4096, 4096), (3, 1023)]  # 含非对齐尺寸
DTYPES = [torch.float32, torch.float16]
LAYOUTS = ["contiguous", "transposed"]


@dataclass
class OperatorCase:
    case_id: str
    shape: tuple
    dtype: torch.dtype
    layout: str

    def make_input(self, seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(*self.shape, generator=g)
        if self.layout == "transposed":
            x = x.T  # 制造 non-contiguous tensor
        return x


def generate_cases(op_name: str) -> list[OperatorCase]:
    cases = []
    for shape, dtype, layout in itertools.product(SHAPES, DTYPES, LAYOUTS):
        case_id = f"{op_name}-{shape[0]}x{shape[1]}-{dtype}-{layout}"
        cases.append(OperatorCase(case_id, shape, dtype, layout))
    return cases
```

```python
# tests/test_operator_matrix.py
import pytest
import torch

from src.case_generator import generate_cases
from src.operators import OPERATORS

TOLERANCE = {torch.float32: dict(atol=1e-6, rtol=1e-5), torch.float16: dict(atol=1e-3, rtol=1e-3)}


@pytest.mark.gpu
@pytest.mark.parametrize(
    "op_name,case",
    [(name, case) for name in OPERATORS for case in generate_cases(name)],
    ids=lambda v: v.case_id if hasattr(v, "case_id") else v,
)
def test_operator_correctness(op_name, case):
    op = OPERATORS[op_name]()
    x = case.make_input()

    ref = op.reference(x)
    cand = op.candidate(x, dtype=case.dtype).cpu().float()

    assert not torch.isnan(cand).any() and not torch.isinf(cand).any(), case.case_id
    torch.testing.assert_close(cand, ref, **TOLERANCE[case.dtype], msg=lambda m: f"{case.case_id}: {m}")
```

**今日测试要点**：确认 non-contiguous 用例真的暴露了 layout 相关 bug（可以故意在 `MatMulOp.candidate` 里漏掉 `.contiguous()` 调用一次，验证测试确实会失败，再改回来）。

---

### 第 3 天：算子梯度测试与边界/负向测试

**前置知识点**：`tensor.requires_grad_()`、`loss.backward()`、`param.grad`、`optimizer.zero_grad()` 的作用。

**今日目标**：为可训练算子（`matmul`、`layernorm`、`gelu`）加梯度测试；为全部算子加边界与异常输入测试。

**验收标准**：
- 梯度测试对比 `.grad`，容差与前向一致；测试前清空梯度，测试后断言无 NaN/Inf。
- 边界测试覆盖：全零输入、极端值（1e4/-1e4）、batch=1、最小合法 shape、错误 dtype、非法 shape，每类都断言具体异常类型或具体数值行为，而不是"程序没崩就算过"。
- 算子层的正确性 + 梯度 + 边界测试合计通过率写入一份 `reports/operator-layer-summary.json`。

**代码骨架**：

```python
# tests/test_operator_gradients.py
import pytest
import torch

from src.operators import OPERATORS

TRAINABLE_OPS = ["matmul", "layernorm", "gelu"]


@pytest.mark.gpu
@pytest.mark.parametrize("op_name", TRAINABLE_OPS)
def test_operator_gradient(op_name):
    op = OPERATORS[op_name]()
    x_ref = torch.randn(128, 128, requires_grad=True)
    x_cand = x_ref.detach().clone().to(device="cuda", dtype=torch.float16).requires_grad_()

    ref_out = op.reference(x_ref)
    ref_out.sum().backward()

    cand_out = op.candidate(x_cand, dtype=torch.float16)
    cand_out.sum().backward()

    assert not torch.isnan(x_cand.grad).any() and not torch.isinf(x_cand.grad).any()
    torch.testing.assert_close(
        x_cand.grad.cpu().float(), x_ref.grad.float(), atol=1e-2, rtol=1e-2
    )
```

```python
# tests/test_operator_boundary.py
import pytest
import torch

from src.operators import OPERATORS


@pytest.mark.gpu
@pytest.mark.parametrize("op_name", OPERATORS.keys())
def test_zero_input_no_nan(op_name):
    op = OPERATORS[op_name]()
    x = torch.zeros(128, 128)
    out = op.candidate(x, dtype=torch.float16)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()


@pytest.mark.gpu
def test_wrong_dtype_raises():
    op = OPERATORS["layernorm"]()
    x = torch.randint(0, 10, (128, 128))  # 错误 dtype：整数
    with pytest.raises((RuntimeError, TypeError)):
        op.candidate(x)
```

**今日测试要点**：至少手动触发一次真实失败（比如故意传入形状不匹配的输入），确认异常信息里包含足够定位信息，而不是笼统的 `RuntimeError`。

---

### 第 4 天：Triton 自定义 Kernel 实现

**前置知识点**：Triton 编程模型——`program_id` 对应哪个数据块、`BLOCK_SIZE` 为什么要是编译期常量、`mask` 如何处理越界。

**今日目标**：用 Triton 实现 Softmax kernel，先跑通正确性。

**验收标准**：
- Triton kernel 输出与 `torch.nn.functional.softmax` 在 FP32/FP16 下的误差在容差内。
- 覆盖全零行、单元素行、大数值行（验证数值稳定性：减最大值再算 exp，避免溢出）。

**代码骨架**：

```python
# triton_kernels/softmax.py
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    numerator = tl.exp(x - x_max)  # 减最大值防止 exp 溢出
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    tl.store(out_ptr + row_start + col_offsets, result, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block_size = triton.next_power_of_2(n_cols)
    _softmax_kernel[(n_rows,)](x, out, n_cols, BLOCK_SIZE=block_size)
    return out
```

```python
# tests/test_triton_kernel.py
import pytest
import torch
import torch.nn.functional as F

from triton_kernels.softmax import triton_softmax

TOLERANCE = {torch.float32: dict(atol=1e-5, rtol=1e-5), torch.float16: dict(atol=1e-3, rtol=1e-3)}


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(4, 128), (8, 4097), (1, 8)])
def test_triton_softmax_correctness(dtype, shape):
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    ref = F.softmax(x, dim=-1)
    out = triton_softmax(x)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()
    torch.testing.assert_close(out, ref, **TOLERANCE[dtype])
```

**今日测试要点**：`(8, 4097)` 这种非 2 的幂次列数，用来验证 `mask` 逻辑是否正确处理了越界访问。

---

### 第 5 天：Triton Kernel 性能对比 + 开始模型层

**前置知识点**：CUDA Event 计时、warmup 的必要性（第一次调用包含 kernel 编译/JIT 开销）。

**今日目标**：给 Triton kernel 加性能测试；开始写 `TinyTransformerEncoder`。

**验收标准**：
- Triton kernel 与 PyTorch 内置 softmax 的延迟对比数据真实采集（不能预设结论），如实记录谁更快。
- `TinyTransformerEncoder` 能在固定种子下产出确定性输出（同样输入跑两次结果一致）。

**代码骨架**：

```python
# tests/test_triton_kernel.py（追加）
def _benchmark(fn, x, n_iters=200, warmup=50):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


@pytest.mark.gpu
def test_triton_softmax_benchmark():
    x = torch.randn(32, 4096, device="cuda", dtype=torch.float16)
    triton_ms = _benchmark(lambda t: triton_softmax(t), x)
    torch_ms = _benchmark(lambda t: F.softmax(t, dim=-1), x)
    print(f"\nTriton: {triton_ms:.4f}ms  PyTorch: {torch_ms:.4f}ms")
    # 不断言谁更快，只记录真实数据到报告
```

```python
# src/model.py
import torch
import torch.nn as nn


class TinyTransformerEncoder(nn.Module):
    def __init__(self, vocab_size=32000, hidden=256, seq_len=512, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.linear1 = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        x = x * mask
        h = self.norm1(x + torch.nn.functional.gelu(self.linear1(x)))
        h = self.norm2(h + torch.nn.functional.softmax(self.linear2(h), dim=-1))
        return h
```

**今日测试要点**：写一个简单的确定性测试——同一输入跑两次 `model.eval()` 前向，断言输出完全相等。

---

### 第 6 天：ONNX 导出与校验

**前置知识点**：`torch.onnx.export` 的 `dynamic_axes`、`opset_version`、`onnx.checker.check_model` 的作用。

**今日目标**：实现 `export_onnx.py`，导出模型并做合规性检查。

**验收标准**：
- 导出的 ONNX 通过 `onnx.checker.check_model`。
- `input_ids`/`attention_mask` 的 batch 和 sequence 维度标记为动态轴。
- 输出文件、模型权重、ONNX 文件三者的 SHA-256 都被记录。

**代码骨架**：

```python
# src/export_onnx.py
import hashlib

import onnx
import torch

from src.model import TinyTransformerEncoder


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def export(onnx_path: str = "model.onnx", seed: int = 0) -> dict:
    model = TinyTransformerEncoder(seed=seed).eval()
    input_ids = torch.randint(0, 32000, (4, 128), dtype=torch.int32)
    attention_mask = torch.ones(4, 128, dtype=torch.int32)

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "hidden_states": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    return {"onnx_sha256": sha256_of_file(onnx_path), "seed": seed}


if __name__ == "__main__":
    print(export())
```

**今日测试要点**：`tests/test_export.py` 里跑一次导出，断言 checker 不抛异常，并断言两次导出的 hash 相同（可复现性）。

---

### 第 7 天：TensorRT FP16 Engine 构建

**前置知识点**：`IOptimizationProfile` 的 min/opt/max、`BuilderFlag.FP16`、为什么 Engine 是针对特定 GPU/TensorRT 版本序列化的。

**今日目标**：实现 `build_engine.py`，把 ONNX 编译为带 optimization profile 的 FP16 Engine。

**验收标准**：
- Engine 构建成功并序列化为 `.plan` 文件。
- Profile 的 min/opt/max 与文档中的 `(1,8)/(4,128)/(8,512)` 一致。
- Engine SHA-256 和 builder 配置写入元数据文件。

**代码骨架**：

```python
# src/build_engine.py
import hashlib

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path: str, engine_path: str) -> dict:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    for name in ("input_ids", "attention_mask"):
        profile.set_shape(name, min=(1, 8), opt=(4, 128), max=(8, 512))
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    engine_hash = hashlib.sha256(serialized_engine).hexdigest()
    return {"engine_sha256": engine_hash, "profile": {"min": (1, 8), "opt": (4, 128), "max": (8, 512)}}


if __name__ == "__main__":
    print(build_engine("model.onnx", "model.plan"))
```

**今日测试要点**：断言用相同 ONNX 和 TensorRT 版本重复构建两次，Engine hash 一致（同一环境下的确定性）；若不一致，记录下来这是已知的 TensorRT 构建非确定性，不要在测试里强行断言相等。

---

### 第 8 天：TensorRT Runner 与端到端正确性

**前置知识点**：`create_execution_context`、`set_input_shape`、`set_tensor_address`、`execute_async_v3`、为什么要保留 tensor 引用直到 stream 同步完成。

**今日目标**：实现 `trt_runner.py`，跑通真实推理并与 PyTorch FP32 参考对比。

**验收标准**：
- 对 profile 内至少 3 组 shape，TensorRT 输出与 PyTorch FP32 参考的 `max_abs_error`、`relative_l2_error` 都被计算并落盘。
- 使用文档中约定的 FP16 容差（`rtol=5e-3, atol=5e-3`）。
- 输出中没有 NaN/Inf。

**代码骨架**：

```python
# src/trt_runner.py
import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TrtRunner:
    def __init__(self, engine_path: str):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def run(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        assert input_ids.is_cuda and attention_mask.is_cuda
        batch, seq = input_ids.shape

        self.context.set_input_shape("input_ids", (batch, seq))
        self.context.set_input_shape("attention_mask", (batch, seq))

        output = torch.empty((batch, seq, 256), device="cuda", dtype=torch.float16)

        # 复用 PyTorch 已分配的 CUDA 地址，避免额外拷贝
        self.context.set_tensor_address("input_ids", input_ids.data_ptr())
        self.context.set_tensor_address("attention_mask", attention_mask.data_ptr())
        self.context.set_tensor_address("hidden_states", output.data_ptr())

        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        # 保留 input_ids/attention_mask/output 引用直到这里，防止提前释放
        return output
```

```python
# tests/test_correctness.py
import pytest
import torch

from src.model import TinyTransformerEncoder
from src.trt_runner import TrtRunner

SHAPES = [(1, 8), (4, 128), (8, 512)]


@pytest.mark.gpu
@pytest.mark.parametrize("batch,seq", SHAPES)
def test_trt_matches_pytorch_reference(batch, seq, trt_runner, seed=0):
    model = TinyTransformerEncoder(seed=seed).eval().cuda()
    input_ids = torch.randint(0, 32000, (batch, seq), device="cuda", dtype=torch.int32)
    attention_mask = torch.ones(batch, seq, device="cuda", dtype=torch.int32)

    with torch.no_grad():
        ref = model(input_ids.long(), attention_mask.float()).float()

    out = trt_runner.run(input_ids, attention_mask).float()

    assert not torch.isnan(out).any() and not torch.isinf(out).any()
    torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)
```

**今日测试要点**：先用一组已知会通过的 shape 验证链路能跑通，再逐步加大 batch/sequence，观察误差是否随 shape 增长而系统性变化（如果是，说明累加顺序或算子融合有问题，需要记录而不是忽略）。

---

### 第 9 天：动态 Shape 与负向测试

**前置知识点**：TensorRT 对 profile 外 shape 的报错方式；损坏文件如何在 `deserialize_cuda_engine` 阶段体现。

**今日目标**：实现 `test_dynamic_shapes.py` 和 `test_negative_cases.py`。

**验收标准**：
- Profile 内非对齐 shape（如 `(3, 127)`）能正确执行。
- Profile 外 shape（`batch=0`、`sequence=513`）、错误 dtype、缺少 tensor name、损坏 Engine 文件分别断言预期的异常类型和清晰错误信息。
- 每个负向用例都有对应的中文/英文注释说明"这里在验证什么失败行为"。

**代码骨架**：

```python
# tests/test_dynamic_shapes.py
import pytest
import torch

IN_PROFILE_UNALIGNED = [(3, 127), (5, 200)]


@pytest.mark.gpu
@pytest.mark.parametrize("batch,seq", IN_PROFILE_UNALIGNED)
def test_in_profile_unaligned_shape_runs(batch, seq, trt_runner):
    input_ids = torch.randint(0, 32000, (batch, seq), device="cuda", dtype=torch.int32)
    attention_mask = torch.ones(batch, seq, device="cuda", dtype=torch.int32)
    out = trt_runner.run(input_ids, attention_mask)
    assert out.shape == (batch, seq, 256)
```

```python
# tests/test_negative_cases.py
import pytest
import tensorrt as trt
import torch


@pytest.mark.gpu
def test_out_of_profile_shape_rejected(trt_runner):
    input_ids = torch.randint(0, 32000, (8, 513), device="cuda", dtype=torch.int32)  # 超出 max=512
    with pytest.raises((RuntimeError, ValueError)):
        trt_runner.context.set_input_shape("input_ids", tuple(input_ids.shape))


def test_corrupted_engine_raises(tmp_path):
    bad_engine = tmp_path / "corrupted.plan"
    bad_engine.write_bytes(b"not a real engine")

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(bad_engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine is None  # TensorRT 对损坏文件返回 None 而不是抛异常，必须显式检查
```

**今日测试要点**：`test_corrupted_engine_raises` 提醒一个常见坑——TensorRT 反序列化失败通常返回 `None` 而非抛异常，代码里必须显式检查，这是文档里"负向测试证明失败行为，而非只断言程序报错"的具体体现。

---

### 第 10 天：性能测试与基线候选生成

**前置知识点**：为什么要区分设备端 kernel latency 和端到端 latency；bootstrap 置信区间的直觉。

**今日目标**：实现 `test_performance.py`，采集原始延迟样本并生成候选基线。

**验收标准**：
- 每种 shape 先 100 次 warmup，再用 CUDA Event 采集 500 次真实推理。
- 原始样本、median、p95、mean、标准差都写入 JSON。
- 首次运行只生成候选基线文件，不自动写入 `baselines/` 正式目录（需要人工审阅后手动批准）。

**代码骨架**：

```python
# tests/test_performance.py
import json
import statistics

import pytest
import torch

SHAPES = [(1, 8), (4, 128), (8, 512)]


def _measure(trt_runner, batch, seq, warmup=100, n_iters=500):
    input_ids = torch.randint(0, 32000, (batch, seq), device="cuda", dtype=torch.int32)
    attention_mask = torch.ones(batch, seq, device="cuda", dtype=torch.int32)

    for _ in range(warmup):
        trt_runner.run(input_ids, attention_mask)
    torch.cuda.synchronize()

    samples = []
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        trt_runner.run(input_ids, attention_mask)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


@pytest.mark.gpu
@pytest.mark.parametrize("batch,seq", SHAPES)
def test_performance_candidate_baseline(batch, seq, trt_runner, tmp_path):
    samples = _measure(trt_runner, batch, seq)
    stats = {
        "median": statistics.median(samples),
        "p95": sorted(samples)[int(len(samples) * 0.95)],
        "mean": statistics.mean(samples),
        "stdev": statistics.stdev(samples),
        "raw_samples": samples,
    }
    out_path = tmp_path / f"candidate-{batch}x{seq}.json"
    out_path.write_text(json.dumps(stats))
    assert stats["median"] > 0  # 至少确认测量本身有效，回归判定留给基线批准后的 CI
```

**今日测试要点**：连续跑 5 次这个测试（5 个独立 trial），确认 median 之间的波动幅度，这是后续判断"退化阈值 1.08 倍是否合理"的真实依据，不能只凭猜测设定。

---

### 第 11 天：稳定性 Soak 测试与资源监测

**前置知识点**：`torch.cuda.mem_get_info()`、`torch.cuda.max_memory_allocated()` 的含义与区别。

**今日目标**：实现 `test_soak.py`，连续执行 1000 次推理并监测资源。

**验收标准**：
- 1000 次循环全部无 CUDA error、无 NaN/Inf、无输出漂移（同输入的输出在循环内保持一致）。
- 循环前后 `free memory` 差异记录在报告里，超过预设容差（如 50MB）标记为"资源增长检测"，而不是直接判定为内存泄漏。

**代码骨架**：

```python
# tests/test_soak.py
import pytest
import torch


@pytest.mark.gpu
@pytest.mark.slow
def test_soak_1000_iterations(trt_runner):
    input_ids = torch.randint(0, 32000, (4, 128), device="cuda", dtype=torch.int32)
    attention_mask = torch.ones(4, 128, device="cuda", dtype=torch.int32)

    free_before, _ = torch.cuda.mem_get_info()
    first_output = None

    for i in range(1000):
        out = trt_runner.run(input_ids, attention_mask)
        assert not torch.isnan(out).any() and not torch.isinf(out).any(), f"iteration {i}"
        if i == 0:
            first_output = out.clone()
        elif i % 100 == 0:
            torch.testing.assert_close(out, first_output, rtol=1e-3, atol=1e-3)

    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    growth_mb = (free_before - free_after) / (1024 ** 2)
    assert growth_mb < 50, f"resource growth detected: {growth_mb:.2f} MB"
```

**今日测试要点**：故意在循环中不清理某个中间 tensor 的引用，验证测试能捕获到显存增长；确认后再改回正确实现，形成一次可展示的"发现资源增长"案例。

---

### 第 12 天：报告系统与 release-summary.md 生成器

**前置知识点**：无新概念，重点是把前 11 天产出的数据结构化成文档中约定的 `summary.json` 和 `release-summary.md` 格式。

**今日目标**：实现 `scripts/run_suite.py`，串联所有测试套件并生成报告。

**验收标准**：
- `summary.json` 包含 git commit、模型/Engine hash、环境 fingerprint、每个 case 的正确性和性能数据。
- `release-summary.md` 能让不懂技术细节的人在 30 秒内看懂本次结论。
- JUnit XML 可以被 CI 系统解析展示。

**代码骨架**：

```python
# scripts/run_suite.py
import json
import subprocess
from datetime import datetime, timezone


def collect_environment() -> dict:
    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {"gpu_name": gpu_name, "git_commit": git_commit}


def generate_release_summary(results: dict, out_path: str = "reports/release-summary.md"):
    status = results["status"]
    lines = [
        f"Release decision: {status}",
        f"Reason: {results.get('reason', 'all checks passed')}",
        f"Correctness: {results['correctness_summary']}",
        f"Performance: {results['performance_summary']}",
        f"Reproduce: {results['reproduce_command']}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    env = collect_environment()
    results = {
        "status": "PASS",
        "correctness_summary": "all in-profile cases passed",
        "performance_summary": "see summary.json for raw samples",
        "reproduce_command": "docker run ... python scripts/run_suite.py",
    }
    summary = {
        "git_commit": env["git_commit"],
        "environment": env,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **results,
    }
    with open("reports/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    generate_release_summary(results)
```

**今日测试要点**：手动制造一次失败（比如临时把某个容差改小），确认 `release-summary.md` 里能正确显示 `BLOCKED` 和失败原因，而不是显示 `PASS`。

---

### 第 13 天：CI 配置与 Docker 打包验证

**前置知识点**：GitHub Actions 的 `self-hosted` runner 标签、`workflow_dispatch`/`schedule` 触发器。

**今日目标**：配置 `.github/workflows/gpu-regression.yml`，验证整个流程能在 CI 里跑通。

**验收标准**：
- PR 触发 CPU smoke（`pytest -m "not gpu"`）。
- 定时任务在 self-hosted GPU runner 上跑完整套件。
- Docker 镜像能在全新机器上 `docker build` + `docker run` 复现同样的报告。

**代码骨架**：

```yaml
# .github/workflows/gpu-regression.yml
name: gpu-regression

on:
  pull_request:
  schedule:
    - cron: "0 18 * * *"  # 每日 UTC 18:00

jobs:
  cpu-smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/ -m "not gpu"

  gpu-full-suite:
    if: github.event_name == 'schedule'
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t trt-regression:ci -f docker/Dockerfile .
      - run: |
          docker run --gpus all --rm -v "$PWD:/workspace" trt-regression:ci \
            python scripts/run_suite.py --suite operator,triton,correctness,dynamic,performance,soak
      - uses: actions/upload-artifact@v4
        with:
          name: reports
          path: reports/
```

**今日测试要点**：在一台没跑过这个项目的机器（或全新容器）上，只靠 README 的命令从零开始跑一遍，记录每一个卡住的地方并修复文档，这是对"可复现"承诺的真实检验。

---

### 第 14 天：端到端演示、README 与简历素材

**今日目标**：录制或截图一次完整的"引入问题 → BLOCKED → 复现 → 修复 → PASS"演示；完成 README；整理简历和面试素材。

**验收标准**：
- 演示步骤：(1) 故意把某个 shape 改到 profile 外，或人为加一次多余的 `torch.cuda.synchronize()`；(2) 跑 `run_suite.py` 得到 `BLOCKED` 或 `PERFORMANCE REVIEW REQUIRED`；(3) 用 `release-summary.md` 里的复现命令复现问题；(4) 撤销改动，确认恢复 `PASS`。
- README 包含：一条从零运行命令、项目边界（只覆盖哪些算子/模型/profile）、术语解释（为什么用 median/p95、为什么区分 NOT COMPARABLE）。
- 简历段落和面试展示要点全部替换成本次 14 天里跑出来的真实数字，不使用预设指标。

**验收清单（对照检查）**：
- [ ] `pytest tests/ -m "not gpu"` 在任意机器上通过
- [ ] `docker run --gpus all ...` 在真实 GPU 上完整跑完 operator + triton + correctness + dynamic + negative + performance + soak
- [ ] `reports/summary.json` 和 `reports/release-summary.md` 都是本次真实运行生成的，没有手工编造的数字
- [ ] 至少有一次自然或人为触发的 `BLOCKED`，并且能复现
- [ ] self-hosted GPU CI 至少跑过一次并留有记录
- [ ] README 里的限制说明与代码实际能力一致（没有夸大覆盖范围）

## 简历写法：只替换为已验证的真实数据

不要使用预设的“提升 35%”或“发现 2 个缺陷”。项目完成后，按实际报告填写：

> 构建 GPU 推理发布质量门禁，覆盖算子级 shape×dtype×layout×device 测试矩阵、梯度验证、自定义 Triton kernel 正确性与性能，以及 TensorRT FP16 Engine 的动态 Shape、异常输入与 1000 次稳定性测试；以 PyTorch CPU/GPU 作为参考，基于 CUDA Event 和版本化基线识别设备端延迟回归。自动生成 PASS/BLOCKED/NOT COMPARABLE 发布摘要，并记录 Engine/模型哈希及 CUDA、TensorRT、GPU 环境指纹，使问题可在固定容器中复现。

在有真实数据后补充一条量化成果，例如：

> 在 `<GPU 型号>`、`<镜像版本>` 上将 `<batch, sequence>` 的 p95 设备端延迟从 `<实测值>` 降至 `<实测值>`；通过回归测试定位 `<真实根因，如 profile 设置或额外同步>`，并新增用例防止复发。

## 面试展示要点

面试时展示一次真实发布摘要和一次故意制造的失败，而不是只讲架构：

1. 展示 FP16 输出为何不能使用 `==`，以及为什么当前容差有实验依据。
2. 解释 PyTorch tensor 地址如何绑定到 TensorRT、为什么 tensor 生命周期和 stream 同步重要。
3. 展示 batch/sequence 超出 optimization profile 时的可预期失败。
4. 展示用 CPU 计时为何会错误、CUDA Event 测量的边界是什么。
5. 演示人为加入一次同步或重复推理后，回归 gate 如何捕获性能退化。
6. 说明哪些性能结果不可比较，例如 GPU 型号、时钟/power limit、镜像或 TensorRT 版本变化。
7. 从发布负责人的角度解释 `PASS`、`BLOCKED` 与 `NOT COMPARABLE` 分别触发什么后续动作。
8. 展示 non-contiguous layout 下算子结果与 contiguous 不一致的一次真实失败，以及为什么内存布局会影响数值结果。
9. 解释 Triton kernel 与 PyTorch 内置实现的正确性验证顺序，为什么必须先通过正确性才能讨论性能。

## 可选第二阶段：Triton 服务契约与负载回归

只有第一阶段完成且目标职位明确涉及服务端推理时，才增加第二个项目。范围改为 **Triton TensorRT-Plan Serving Contract and Load Regression**，不假设原生 SSE 或未定义的 `/generate_stream` 接口。

使用 Triton 标准 v2 HTTP 或 gRPC inference protocol 部署同一个 TensorRT Plan 模型，测试内容为：模型仓库加载、输入/输出 schema、batching 配置、并发请求的正确性、HTTP/gRPC 错误码、成功率、服务端队列指标和端到端延迟。动态 batching 参数只在对应 backend 支持且 `config.pbtxt` 有明确配置时测试。

若目标是 LLM 流式岗位，则必须单独选定 TensorRT-LLM backend 或 OpenAI-compatible frontend，并先把“有效 token event”的协议定义、token 数和流式错误语义写成契约。此时才可测量：

$$
TTFT = t_{first\ valid\ token} - t_{request\ start}
$$

$$
TPOT = \frac{t_{last\ token} - t_{first\ valid\ token}}
{N_{output\ tokens} - 1}
$$

每个负载配置应有 warmup、足够的完成请求量、重复 trial、错误率和原始时序数据；少量并发请求的最大值不能包装成可靠的 p99。

## 完成定义

项目完成不是文档写完，而是同时具备以下证据：真实 Docker 环境、可执行代码、算子级测试矩阵的真实误差报告、至少一个通过验证的 Triton kernel、一份 `release-summary.md`、至少一次 GPU CI 报告、批准的性能基线、一个可重复的负向失败用例，以及一份能解释环境与统计方法的 README。

最终作品集应呈现一个完整演示：修改一个会导致输入契约失败或性能退化的配置，运行测试得到 `BLOCKED` 或 `PERFORMANCE REVIEW REQUIRED`，从摘要复制命令复现，再修复并得到 `PASS`。达到这些条件后，它可以作为 NVIDIA 推理 QA、GPU 软件测试、AI 基础设施测试开发及 GPU 创业公司性能测试岗位的有效作品集。
