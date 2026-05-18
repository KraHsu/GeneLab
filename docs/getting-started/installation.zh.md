# 安装

这页是专门的安装检查清单。完整学习路径请从[教程](../tutorial.md)开始。

## 要求

| 要求 | 版本 |
|---|---|
| Python | 3.12 或更新 |
| 依赖管理 | `uv` |
| 仿真后端 | `genesis-world>=0.4.7` |
| PyTorch | 通过一个 `torch-*` extra 安装 `torch>=2.8.0` |

## 同步环境

在仓库根目录：

```bash
uv sync --extra torch-cpu
uv run genelab --version
```

只能选择一个 PyTorch extra：

| Extra | 目标 |
|---|---|
| `torch-cpu` | CPU-only 或非 NVIDIA 开发机器。 |
| `torch-cu126` | 兼容 CUDA 12.6 wheel 的 NVIDIA 驱动。 |
| `torch-cu128` | 兼容 CUDA 12.8 wheel 的 NVIDIA 驱动。 |
| `torch-cu130` | 兼容 CUDA 13.0 wheel 的 NVIDIA 驱动。 |

刷新已有 torch：

```bash
uv sync --reinstall-package torch --extra torch-cu128
```

## 创建本地缓存

```bash
uv run genelab cache
```

该命令创建项目本地可写缓存目录，并把 `XDG_CACHE_HOME` 和 `MPLCONFIGDIR` 指向 `.cache/`。

## 验证导入

```bash
uv run python -c "import genelab; print(genelab.__version__)"
uv run python -c "from genelab.lab import TaskCfg; print(TaskCfg.__name__)"
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
uv run python -c "import genesis; print(genesis.__version__)"
```

## 安装示例扩展

核心包是框架；任务由扩展注册。

```bash
uv pip install -e examples/inverted_pendulum
uv pip install -e examples/genelab_examples
uv run genelab list tasks
```

只有需要更大的人形机器人示例时再安装 Unitree：

```bash
uv pip install -e examples/unitree
```

## 另见

- [教程](../tutorial.md)
- [调试常见问题](../best-practices/debugging.md)
- [构建扩展项目](../best-practices/extension-projects.md)
