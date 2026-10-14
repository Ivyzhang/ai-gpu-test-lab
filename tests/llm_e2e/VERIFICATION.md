# vLLM 迁移验证记录

日期：2026-10-15。

## 变更范围

- 服务启动器改为 `vllm serve`，读取完整 CLI 帮助 `--help=all`。
- 单 GPU，固定服务模型名，max-model-len=4096、max-num-seqs=8、TP=1、dtype=auto、GPU 内存预算比例 0.8。
- 记录 vLLM 版本和 runtime 身份；拒绝旧部署 manifest，重新建立 vLLM 基线。
- HTTP/SSE、并发、TTFT/E2E/TPOT 估计、吞吐和门禁继续使用现有测试逻辑。
- README 更新为 vLLM 部署步骤。

## 本机验证

从项目根目录执行 `python3 -B -m unittest discover -s tests/llm_e2e/tests -v`，结果：**Ran 52 tests in 2.001s — OK**，退出码 0。两个 CLI 的 `--help` 检查通过；完整输出保存到 `LOCAL_TEST_RESULTS.txt`。

本机 HTTP 测试仅监听 127.0.0.1 临时端口。vLLM 启动参数和预检测试使用模拟 CLI/GPU 查询；没有加载真实模型。

## 未验证

本机没有可用于本项目的 NVIDIA GPU/vLLM 服务。没有执行真实模型部署、GPU 并发或性能验收。官方文档联网查询失败，启动器会在目标机器按安装版本帮助检查参数，再通过真实生成探针验证服务；最终兼容性需在目标 GPU 验证。

本次本机环境：macOS-26.6.2-x86_64-i386-64bit-Mach-O，Python 3.13.5。vllm 命令可用：False；nvidia-smi 可用：False。
