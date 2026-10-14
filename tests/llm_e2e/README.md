# Qwen2.5-0.5B-Instruct：vLLM 服务端到端测试

这套测试把原项目的 DistilBERT Encoder 路径扩展为独立的生成式服务验证路径：**本地模型 → vLLM Server → 真实 HTTP 请求 → 流式响应 → 并发指标 → 报告和门禁**。

选用 `Qwen/Qwen2.5-0.5B-Instruct`。约 5 亿参数，适合部署冒烟、协议和并发验证；小模型的通过结果不能代表大型模型的性能或业务效果。FP16/BF16 权重体积约为参数量乘 2 字节，实际显存还包含 Runtime、工作区、KV Cache 和批处理开销，不能据此保证特定显存容量可运行。

## 1. 当前交付及验证边界

- 已提供：服务启动器、模型内容 hash、CLI 兼容检查、HTTP/SSE 客户端、功能测试、1/2/4/8 并发、短/长/混合请求、逐请求记录、延迟和吞吐、门禁、GPU 资源采样、中文报告。
- 原来的 DistilBERT 数值测试仍有价值，但它不生成 token，不适用 TTFT/TPOT。这里没有把 DistilBERT 改名为生成模型。
- 本套启动器使用 **vLLM 的 OpenAI 兼容服务**，入口为 `vllm serve`，直接加载本地 Hugging Face 模型权重。该服务测试不依赖原项目的 TensorRT Engine 构建/Runner；原有 Encoder 和算子测试仍独立存在。
- 本机 CPU 协议测试使用模拟 HTTP 服务，检查的是测试工具，不证明真实模型/GPU 通过。
- 本次官方文档在线核对失败。以下部署命令带运行时 `--help` 检查，但 CLI 检查不能替代模型、GPU、驱动和所选 vLLM 版本的实机兼容验证。没有捏造已验证版本号或容器 tag。
- 本模块位于项目根目录下的 `tests/llm_e2e/`，独立于已有的 `src/` 和 DistilBERT 测试。以下命令均在 `/Users/ivyzhang/Desktop/workspace/ai-gpu-test-lab/` 项目根目录执行；复制到 GPU 服务器后，在对应项目根目录执行。

## 2. 文件说明

```text
tests/llm_e2e/
  serve.py               本地模型校验、启动、信号转发、GPU CSV
  client.py              真实 HTTP/SSE、总请求期限、错误和时间戳
  deployment.py          本机监听端口与服务进程归属校验
  run.py                 功能、负向恢复、固定并发、原始报告
  metrics.py             分位数、吞吐、绝对 SLA、相对回归门禁
  config.example.json    冒烟规模、未填写的业务 SLA
  tests/
    test_harness.py       计时/门禁/线程并发逻辑测试
    test_serve.py         模型身份和启动参数检查
    test_http.py          本机 HTTP 协议集成测试，无 GPU 模型
    test_deployment.py    预检/停止状态及错误进程、地址归属回归
  README.md              本说明
```

客户端和本机自测仅需要 Python 3.10+ 标准库。服务端需要由团队准备好兼容的 Linux/NVIDIA 环境、vLLM 和模型文件；不要为了运行此客户端而在 macOS 上安装 GPU Runtime。

## 3. 测试对象、输入和判定

### 3.1 功能用例

| 用例 | 步骤 | 判定 | 能说明什么 |
|---|---|---|---|
| 模型可见性 | 轮询 `/v1/models`，检查模型 ID 与部署 manifest | ID 匹配且后续生成成功 | 模型可被访问；models 返回 200 单独不代表推理就绪 |
| non-stream | 调用 chat completions | 内容非空、finish_reason 为 stop/length | 非流式基本可用 |
| stream | 消费 role、内容、usage、DONE | 有内容、合法终止和 DONE | 流式协议完整 |
| Unicode | 中文请求 | 可解码且返回非空内容 | 编码传输，不把它当中文能力评测 |
| max_tokens=1 | 请求计数任务 | usage 恰为 1 且协议完成 | 输出上限生效 |
| 错误模型名 | 请求不存在的模型 | 400/404/422 | 明确拒绝；500 不算正确拒绝 |
| 非法长度参数 | max_tokens=-1 | 400/422 | 参数校验 |
| 错误后恢复 | 再发正常请求 | 正常完成 | 非法请求未破坏服务 |
| 并发算术 | 同时请求 8 个不同加法题，4 并发 | 每条响应恰为对应整数，completion ID 唯一 | 辅助检查响应归属与基础任务执行 |

算术用例失败并不直接证明 Runtime 串请求：小模型可能多输出了解释或算错。必须结合保存的请求索引、目标数值和响应排查。不要为了通过而悄悄放宽业务断言。

### 3.2 性能矩阵

默认 3 个场景 × 4 个并发等级 = 12 组。每组 2 次预热，随后 32 个正式请求，共 384 个正式请求。另有功能请求。

| 场景 | 输入 | 输出上限 | 用途 |
|---|---|---:|---|
| short | 简短英文操作说明请求 | 64 | 基本 TTFT 与端到端响应 |
| long | 160 行日志及摘要要求 | 64 | 较长输入的 prefill 开销 |
| mixed | short/long 交替 | 64 | 混合长度调度；另报告长短子组 |

并发等级为 1、2、4、8。线程池让多个 HTTP 请求真实重叠；每个线程同时最多一个请求，总并发不超过配置值。客户端并发 8 不等于服务端 batch 必定为 8，也不等于 8 张 GPU 或 TP=8。

输入是文本场景，不冒充精确的 128/4096 token：chat template 和 tokenizer 会改变真实长度。正式报告保存 `prompt_tokens`、`completion_tokens` 分布，`max_tokens` 只是上限，允许自然 EOS。要做严格固定 ISL/OSL 的引擎基准，需要另外设计支持该能力的专用 benchmark。

本矩阵采用 **closed-loop 固定并发**。线程完成一个请求后开始下一条；它能比较固定并发下性能，不能直接声称验证了固定到达率或生产峰值 QPS。压测服务越慢，closed-loop 实际发送率越低。

### 3.3 缓存、计时与连接条件

- 每组预热不计入正式样本；启动和模型加载不计入稳态 TTFT。
- 相似 prompt 会共享前缀，本版本明确作为 warm/shared-prefix 场景。记录服务器默认 cache 策略，但未控制 prefix cache 的实际命中率。不能称为 cold-cache 性能。
- 每条请求创建新 HTTP 连接；计时包含本次 TCP/TLS 建连及 HTTP、服务排队、推理和响应传输。生产连接池场景应单独测量，不能把两种口径直接比较。
- 当前整套运行器限定服务和客户端位于同一主机/同一容器，通过 `127.0.0.1` HTTP 连接；测试前后核验监听端口属于此次启动的服务进程组。这样可以阻止旧服务冒用新部署 manifest。生产网关、远端网络、IPv6 暂不纳入此版本。
- 客户端拒绝需要 DNS 查询的主机名（localhost 直接映射 127.0.0.1），避免 DNS 阻塞超出请求期限。服务身份校验不允许用远端数值 IP 绕过本机范围。
- 使用单调时钟计算延迟；原始请求带 started_unix_s 和 wall_ms。GPU CSV 使用设备工具的墙钟时间。

## 4. 指标定义

| 指标 | 算法/口径 |
|---|---|
| 客户端 TTFT | 首个非空 `delta.content` 到达时间 − 请求开始时间；role/空内容不算 |
| E2E | 收到 `[DONE]` 的时间 − 请求开始时间 |
| last_content_ms | 最后一个内容块到达时间 − 请求开始时间 |
| TPOT estimate | `(最后内容到达 − 首内容到达)/(completion_tokens−1)`；N<=1、缺 usage 或只有一个内容块时为 null |
| chunk_gaps_ms | 相邻非空 SSE 内容块到达间隔，不是 token ITL |
| output_tokens_per_s | 全部请求成功且 usage 完整时，总输出 token / 整组墙钟时间；否则 null |
| successful_output_tokens_per_s | 有效成功样本的输出 token / 整组墙钟时间，明确为部分结果 |
| successful_requests_per_s | 成功请求数 / 整组墙钟时间 |
| success_rate | 成功数 / 所有已提交请求；失败、超时不从分母删除 |
| p50/p95/p99 | nearest-rank，分别从成功请求原始延迟计算；同时展示失败率 |

**SSE chunk 不是 token。** 一个 chunk 可能含多个 token，所以 TPOT 只能是客户端估计值，不能用来声称测到了 GPU decode 单步时间。若全部输出在一个 chunk 返回，TPOT 标为不可测（null）；报告另记录 `one_chunk_responses`。TTFT、E2E、TPOT 各自都要达到 min_samples 有效样本，不能用单 token 请求或不可测 TPOT 凑数。正式 ITL 需要服务端逐 token 埋点或经过校准的专用工具。

32 个样本仅适合冒烟和探索；其 p99 通常接近最大值。发布测试请按风险、时长与统计要求增加到数百/数千请求，并做多个独立运行窗口。代码的回归阈值属于确定性启发式，没有实施统计显著性检验或置信区间。

## 5. 在目标 GPU 环境部署

### 5.1 准备模型和软件

由团队根据 vLLM 官方安装说明选择匹配目标 GPU/驱动的 vLLM 版本或容器。不要使用未固定的 `latest` 做可追溯发布验收。GPU 架构、驱动、CUDA 和 vLLM/PyTorch 版本都要匹配；参数量小不代表任意旧 GPU 都被支持。

需要已有的命令：

```sh
nvidia-smi
vllm serve --help=all
python3 --version
```

准备一个完整的本地模型快照，例如 `/models/Qwen2.5-0.5B-Instruct`，包含 config、tokenizer、chat template（可能位于 tokenizer_config.json）和 safetensors 权重。建议固定 Hugging Face commit，而不是浮动 main。

如果团队使用 Hugging Face CLI，可以在**已允许下载的目标服务器**手动运行以下命令；本工具不会执行下载。`MODEL_REVISION` 必须是你审查过的实际模型提交 SHA：

```sh
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --revision "${MODEL_REVISION:?set the approved immutable model commit}" \
  --local-dir /models/Qwen2.5-0.5B-Instruct
```

不同版本 CLI 以安装版本帮助为准。文件 hash 会覆盖权重、配置、tokenizer，补足本地目录没有 revision 字段时的内容身份追溯。哈希检查不是 safetensors 内容可加载性验证，真正的加载错误会出现在 server.log。

### 5.2 启动服务：终端 A

在包含 `tests/llm_e2e/` 的项目根目录执行：

```sh
python3 -m tests.llm_e2e.serve \
  --model-dir /models/Qwen2.5-0.5B-Instruct \
  --cuda-device 0 \
  --output reports/qwen-server-001
```

启动器检查 CLI 中存在所需参数，然后启动的逻辑命令为：

```sh
vllm serve /models/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --served-model-name Qwen/Qwen2.5-0.5B-Instruct \
  --tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 8 \
  --generation-config vllm --dtype auto --gpu-memory-utilization 0.8
```

`CUDA_VISIBLE_DEVICES` 被设置为指定的单张卡。启动器不下载模型、不安装依赖，不保证任意 vLLM 版本都支持该命令；不匹配时会明确停止并保留 `cli-help.txt`。`--dtype auto` 由 vLLM 根据模型配置选择精度，不能宣称强制了 FP16。`--generation-config vllm` 使用服务默认生成配置，避免模型快照中的生成配置引入未声明默认参数；请求仍明确发送 temperature=0 和 max_tokens。GPU 内存预算比例为 0.8，它不表示实际使用量固定为 80%，也不保证不会 OOM。prefix cache 策略仍采用安装版本默认值。

启动器先拒绝已被占用的端口。随后 manifest 从 preflight → starting，只有监听进程归属和一次真实生成都成功后才发布 ready，并打印 `READY`。**等到 READY 再启动终端 B。** 进程退出会把 manifest 改为 stopped/failed；这些状态和 preflight 都不能用于模型验收。Linux 使用 `/proc` 核验监听 socket，客户端需和服务在同一 PID/网络命名空间并具有检查该进程的权限。

使用 `--preflight-only` 仅检查文件、GPU 信息和 CLI，不启动模型，不等于模型验证通过。输出目录必须不存在；每次创建新目录保存证据。

服务日志和资源输出：

```text
reports/qwen-server-001/
  deployment.json         模型 hash、GPU、软件版本、启动命令
  cli-help.txt            实际安装版本的命令帮助
  server.log              模型加载、推理服务 stdout/stderr
  startup-probe.json      启动后的真实生成探针结果
  gpu.csv                 约每秒采集设备利用率、显存、功耗、温度
  telemetry-errors.log    采样失败时才生成
  server-exit.json        服务退出后生成
```

GPU CSV 是设备级监测，可能包含其他进程；它不是 KV Cache 使用率或进程显存泄漏证明。当前只采集这些资源指标，尚未为资源趋势设置自动门禁。

### 5.3 运行客户端：终端 B

先保持示例 SLA 为 null，做一次探索运行：

```sh
python3 -m tests.llm_e2e.run \
  --config tests/llm_e2e/config.example.json \
  --deployment reports/qwen-server-001/deployment.json \
  --output reports/qwen-exploration-001
```

如果模型和测试正常、仅 SLA 未配置，结果为 `NEEDS_CONFIGURATION`，退出码 2。这是正确行为，不是工具故障。若缺 usage、功能失败或样本缺失，则优先报告 BLOCKED。

启动器使用 `--served-model-name Qwen/Qwen2.5-0.5B-Instruct` 固定 API 模型名，manifest 只允许该名字。配置里的 `served_model_id` 保持 null 即可自动选择唯一模型，或填写该固定名字。旧部署 manifest 不含 vLLM 身份，运行器会拒绝使用；应通过新启动器生成新的 manifest。

服务固定绑定 IPv4 127.0.0.1，允许修改端口，但客户端配置的端口必须与 manifest 一致。当前不支持远端压测或网关认证链路；这类链路需要另外增加可验证的服务身份机制。如果本机服务开启鉴权，可设置 `LLM_API_KEY` 环境变量；报告不保存密钥。

### 5.4 设置真实门禁

复制配置为自己的配置文件，修改：

- `requests_per_case` 和 `min_samples`：两者不应少于你确定的验收样本量。
- `ttft_p95_ms`：用户侧首内容延迟上限。
- `e2e_p95_ms`：完整协议响应上限。
- `tpot_estimate_p95_ms`：已接受该估计口径后的上限。
- `max_regression_fraction`：例子为 0.10（10%），仅供启动工程流程，须结合噪声和业务批准。

本版所有场景共享一组绝对 SLA；如果不同输入长度有不同 SLA，应先按场景拆成多个配置运行。混合场景的长短子组会单独报告，但没有独立子组门禁，因此不能仅靠总 p95 宣称短请求无饥饿。

配置好后，用新目录重新执行。绝对门禁全部通过的报告仍会标记 `baseline_compared=false`，它不代表已完成版本回归比较。

### 5.5 回归比较

由负责人审核一份真实 GPU 的 PASS 报告，保留只读副本作为 baseline。工具不自动把新结果覆盖为基线。例：

```sh
python3 -m tests.llm_e2e.run \
  --config tests/llm_e2e/config.gpu.json \
  --deployment reports/qwen-server-002/deployment.json \
  --baseline reports/qwen-approved-001/summary.json \
  --require-baseline \
  --output reports/qwen-regression-002
```

其中 `config.gpu.json` 是你根据 SLA 创建的配置，`qwen-approved-001` 是人工审查后保留的真实报告目录；它们不是本包预置的虚构结果。

比较要求：报告 schema 兼容、baseline 原门禁 PASS、case 集合一致、模型 hash/GPU/启动参数/请求场景等受控条件一致、baseline 样本和 usage 完整。模型绝对目录也是当前启动命令的一部分，路径变化会导致 NOT_COMPARABLE；需要比较时使用相同挂载路径。

vLLM runtime 身份、启动参数保留在 controlled 中；旧引擎基线不能直接当作此次 vLLM 回归基线。请先生成并审核一份 vLLM 基线。软件/驱动版本保留在 manifest 中，不强制相等，允许测试 Runtime 升级。但操作人员必须声明本次改变的变量并检查没有其他环境变化。相同 manifest 不是机器完全无干扰的证明。

## 6. 门禁规则和退出码

| 状态 | 条件 | 退出码 |
|---|---|---:|
| PASS | 必需用例、有效样本、usage、绝对 SLA、可选基线比较全部满足 | 0 |
| BLOCKED | 功能失败、预检失败、缺 case/usage、延迟超限、性能回退超限 | 2 |
| NEEDS_CONFIGURATION | 结果完整但业务 SLA 未配置/无效 | 2 |
| NOT_COMPARABLE | 要求基线却未提供，或基线/环境不满足比较条件 | 2 |

基础设施失败记录在 `infrastructure_error` 并阻断，当前未单独实现 INFRA_FAILURE 状态。客户端配置错误或输出目录已存在时直接退出 2，不覆盖历史结果。

相对比较包括 TTFT p95、E2E p95、TPOT estimate p95、输出 token 吞吐。延迟上升/吞吐下降超过批准比例时阻断。没有置信区间，重复测试前要制定重跑规则并保留首败，不能只挑通过的一次。

报告目录包含：

```text
config.json              本次输入配置
deployment.json          部署 manifest 副本
requests.jsonl           功能、预热和正式请求原始数据，带 phase/case
partial-summary.json     完成一组后保存的中间状态
summary.json             最终结构化结果、门禁和原因
report.md                中文结果表格
```

`requests.jsonl` 包含生成文本，使用公司业务数据时按业务数据的权限和保留策略处理。

## 7. CPU 自测与 CI

不需要 vLLM 或模型即可验证测试工具：

```sh
python3 -m unittest discover -s tests/llm_e2e/tests -v
python3 -m tests.llm_e2e.run --help
python3 -m tests.llm_e2e.serve --help
```

HTTP 集成测试会在 `127.0.0.1` 临时端口启动模拟服务；受限沙箱可能需要放开本机端口权限。它不连接外部服务、不下载模型、不产生真实模型性能基线。

CI 推荐分层：

1. PR CPU：运行以上自测，验证工具逻辑。
2. GPU smoke：目标服务器启动本地模型，用小规模矩阵验证接口和并发；此阶段不宣称 p99 SLA。
3. Nightly GPU：增加请求数、独立运行窗口和场景，使用固定环境/批准基线。
4. Release：必须配置业务 SLA、使用 `--require-baseline`，并补齐下节业务范围要求。

CI 必须保留客户端退出码，失败时也上传客户端与服务器两个完整目录（GitHub Actions 使用 `if: always()`）。不要使用 `|| true` 把非 PASS 吞掉。每个 GPU job 需独占或按团队租约隔离目标设备，结束后停止自己启动的服务；不要把别人的服务当成本次产物。

## 8. 尚未实施，不能宣称已经覆盖

| 项目 | 目前状态 | 后续设计 |
|---|---|---|
| 固定到达率/过载 | 当前 closed-loop | 增加 open-loop 发送器及排队/丢弃统计 |
| 远端网关/IPv6 | 当前本机 IPv4 + 进程归属 | 增加服务实例身份/部署证明，再接入网关和远端压测 |
| 取消后 KV Cache 回收 | 未测 | 客户端断开后检查服务端活跃请求/缓存恢复 |
| KV Cache 资源门禁 | 只有 GPU 总显存 CSV | 接服务端指标，按 cache/block 状态验证 |
| 长稳泄漏 | 目前有限请求数 | 小时级运行、同生命周期采样、平台/趋势判定 |
| 进程故障恢复 | 未测 | 由编排层重启，验证在途错误和恢复时间 |
| 多卡 TP/PP | 单 GPU 服务 | 在支持的硬件和模型上独立建立多卡矩阵 |
| 精度/量化效果 | 没有 logits/golden 对比 | 接固定任务集与参考后端，验证量化误差和任务损失 |
| 完整上下文边界 | 长文本不等于上限 | 目标 tokenizer 精确构造 max−1/max/max+1 |
| Stop、多轮、工具调用 | 未完整测 | 根据产品接口契约增加独立用例 |
| 真正 token ITL | 无逐 token 埋点 | 对接服务端时间戳并与客户端观测关联 |

因此当前交付是一个可执行的**单卡生成服务端到端测试起点**，不是所有企业发布维度都已完成的声明。

## 9. 故障定位

| 现象 | 优先检查 |
|---|---|
| 找不到 vllm serve | 是否进入了准备好的 vLLM 环境/容器 |
| CLI 参数检查失败 | 查看 cli-help.txt，核对安装版本，调整 launcher 适配并重新自测 |
| 模型加载失败 | config 的架构、完整权重、GPU 支持、精度和依赖；查看 server.log |
| readiness 超时 | 监听地址/端口、容器端口映射、模型是否正在初始化、模型 ID |
| usage 缺失 | 所选服务版本是否支持 stream_options.include_usage；不能用 chunk 数替代 |
| 算术失败 | 查看原始文本，区分模型指令遵循与并发响应错配 |
| 只有并发组变慢 | 排队、连续批处理、KV Cache、CPU/客户端瓶颈；不要直接归因 GPU kernel |
| 性能 NOT_COMPARABLE | 模型 hash、GPU UUID、挂载路径、启动配置、场景和样本规模是否变化 |
| GPU 利用率低但 TTFT 高 | 排队、CPU tokenizer、网络、调度和连接成本 |

官方参考入口（本次未成功在线核验，执行前以目标版本为准）：

- [vLLM CLI 文档](https://docs.vllm.ai/en/latest/cli/serve/)
- [vLLM OpenAI 兼容服务](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
- [vLLM 官方代码仓库](https://github.com/vllm-project/vllm)
- [Qwen2.5-0.5B-Instruct 模型](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

## 10. 当前工作区快速打开

完整代码和说明位于本 README 所在的 `tests/llm_e2e/`。以项目根目录运行 `python3 -m ...`，不要直接执行 `python3 tests/llm_e2e/run.py`，后者会失去包的相对导入上下文。

## 11. 在项目 tests 目录运行

模块位置为 `tests/llm_e2e/`，自测用例位于 `tests/llm_e2e/tests/`。从项目根目录执行：

```sh
python3 -m unittest discover -s tests/llm_e2e/tests -v
python3 -m tests.llm_e2e.serve --help
python3 -m tests.llm_e2e.run --help
```

`tests/__init__.py` 只声明本项目测试包，避免与第三方同名 tests 包冲突。以上 unittest 命令不加载已有的 `tests/conftest.py`，因此工具自测仍不要求 PyTorch/Transformers。原项目 pytest 配置和已有 GPU 测试保持原来的运行方式。
