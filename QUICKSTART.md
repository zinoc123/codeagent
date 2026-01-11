# AI Code Agent - 快速开始与测试指南

本文档指导如何在本地环境中启动 AI Code Agent 并进行功能测试。

## 1. 前置条件

确保你的开发机满足以下要求：
- **操作系统**: Linux (推荐 Ubuntu) 或 macOS
- **Docker**: 已安装并运行
- **NVIDIA GPU**: 推荐拥有至少 8GB VRAM 的显卡（用于运行 7B 模型）
- **NVIDIA Container Toolkit**: 确保 Docker 可以使用 GPU (`--gpus all`)

## 2. 启动服务

我们提供了一键启动脚本，它会自动处理以下任务：
1. 创建并激活 Python 虚拟环境 (venv)
2. 安装 Python 依赖
3. 启动 **Qdrant** (向量数据库) Docker 容器
4. 启动 **Ollama** (本地大模型推理) Docker 容器
5. 下载所需的 AI 模型 (`qwen2.5-coder:7b`, `nomic-embed-text`)
6. 启动 **FastAPI** 后端服务

### 启动命令

```bash
bash start.sh
```

**注意**: 首次运行需要下载约 5GB 的模型文件，请耐心等待。看到 `Application startup complete` 日志即表示服务就绪。

## 3. 进行测试

### 准备测试数据
确保 `example1` 目录已打包为 `example1.zip`：
```bash
zip -r example1.zip example1
```

### 运行测试脚本
在另一个终端窗口中运行：

```bash
python3 test_script.py
```

该脚本会自动：
1. 检查 API 是否存活
2. 发送包含 `example1.zip` 和需求描述的 POST 请求
3. 打印分析报告 (JSON)

## 4. 查看结果

成功的响应将包含：
- **feature_analysis**: 针对每个需求点的代码位置分析
- **functional_verification**: 自动生成的测试代码及其执行日志（`tests_passed: true/false`）

---

# 常见问题排查

- **Ollama 下载慢**: 请确保网络畅通，或手动执行 `docker exec ollama_local ollama pull qwen2.5-coder:7b`。
- **显存不足**: 如果 GPU 显存不足，可以尝试更换更小的模型（如 `qwen2.5-coder:1.5b`），需修改 `app/core/agent.py` 中的模型名称。
