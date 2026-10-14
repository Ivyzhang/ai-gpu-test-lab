# 本机验证记录

日期：2026-10-14。

环境：macOS 26.6.2，Python 3.13.5。`trtllm-serve`、`nvidia-smi` 和 `tensorrt_llm` 均不可用。

## 实际执行

```sh
python3 -m unittest discover -s tests/llm_e2e/tests -v
```

最终结果：**Ran 47 tests in 2.323s — OK**，退出码 0。

覆盖：

- SSE 首个 role/空 chunk 不计入 TTFT；内容、usage、DONE 的解析。
- 单 token、单 chunk、缺 usage 不伪造 TPOT。
- HTTP 错误、流中断、请求期限与非流式接口。
- 线程请求实际重叠且不超过配置的并发上限，worker 异常保留为失败。
- 样本数、各指标有效样本数、SLA 缺失、NaN、基线状态、schema、环境和回退门禁。
- 本地模型文件身份随权重/tokenizer 改变，缺权重/错误架构被拒绝。
- 服务预检/停止状态、错误进程组、IPv4/IPv6 归属混淆、已占用端口。
- 本机真实 HTTP 模拟服务驱动功能测试、4 组并发场景、原始请求和报告生成；模拟服务无 GPU、无真实模型。

CLI `--help` 和 Python 模块编译检查也已执行。

本机 HTTP 测试仅绑定 127.0.0.1 临时端口，受沙箱限制后经授权执行。模型文件身份测试使用明确的测试 fixture 文件；不表示测试 fixture 是可加载的模型。

## 审查修复

独立只读审查发现并修复了以下问题：

1. 旧服务可能被误认为新部署：增加端口占用预检、进程/监听 socket 归属、ready 状态与真实生成探针；运行前后复验归属。
2. TPOT 只有少量有效样本仍可能通过：逐指标校验最小有效样本，baseline 同样检查。
3. DNS 可能超出请求期限：拒绝 DNS 主机名，支持范围明确为本机 IPv4 服务。
4. IPv6 的同端口进程可能错误证明 IPv4 服务归属：将服务和归属检查限定到 127.0.0.1/IPv4，并补回归。

## 未验证

- Qwen2.5-0.5B-Instruct 在某个具体 TensorRT-LLM/GPU/驱动组合中的模型加载。
- 真实生成内容、GPU 并发调度、TTFT/TPOT/吞吐数值与模型效果。
- Linux 目标环境中的实际进程归属、服务启动/退出和 GPU CSV 采样；macOS 已验证真实监听端口归属路径。
- 官方文档在线核验（网络查询失败）；目标安装版本必须通过 CLI 预检及实际启动验证。

不能把本记录作为 GPU 发布准入报告。真实部署需要按照 README 启动服务并生成独立 reports 目录。

## 指定项目目录复验

已复制到 `/Users/ivyzhang/Desktop/workspace/ai-gpu-test-lab/tests/llm_e2e/`，并校验了全部复制文件的 SHA-256。随后从该项目根目录重新执行全部 47 项自测，结果通过（退出码 0）；两个 CLI 的 `--help` 检查通过。完整输出见同目录 `LOCAL_TEST_RESULTS.txt`。本次为目录落地与客户端自测，不是 GPU 模型运行验收。

## tests 目录布局复验

模块已移动到 `/Users/ivyzhang/Desktop/workspace/ai-gpu-test-lab/tests/llm_e2e/`，原根目录 `llm_e2e/` 已移走。导入路径更新为 `tests.llm_e2e`，默认配置从模块所在目录读取。已从项目根目录执行 `python3 -m unittest discover -s tests/llm_e2e/tests -v`，全部 47 项通过（退出码 0）；两个新入口的 `--help` 检查通过。最新输出见 `LOCAL_TEST_RESULTS.txt`；未运行真实 GPU 模型。
