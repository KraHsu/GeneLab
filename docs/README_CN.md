# GeneLab

> **本文已迁移至 [文档站点（中文）](https://krahsu.github.io/GeneLab/zh/)。**
> 完整中文文档（CLI 参考、核心概念、API 自动生成参考）请见站点；本文件保留以兼容旧链接。

GeneLab 是一个面向强化学习与机器人研究的 Isaac Lab 风格 API，由
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 提供仿真后端。它保留了机器人、
环境、任务注册，manager-based MDP 配置，以及 CLI 调度这些常见组织方式。

## 目标

- 提供小型机器人、环境和任务注册表。
- 将核心 API 层与示例资产、演示脚本分离。
- 使用 manager 风格配置钩子组织 actions、observations、rewards、events 和 terminations。
- 保持 Genesis 后端集成显式，便于扩展。
- 通过稳定的包结构和 CLI 支持下游机器人研究项目。

## 要求

- Python 3.12 或更新版本。
- 使用 [uv](https://docs.astral.sh/uv/) 管理依赖。

## 设置

在仓库根目录运行：

```bash
uv sync
uv run genelab --help
```

`uv sync` 会创建项目虚拟环境，从当前 checkout 安装 GeneLab，并安装 `uv.lock` 锁定的依赖。
`uv run ...` 会在该环境中运行命令。裸 `genelab` 命令只有在已激活 `.venv`，或 GeneLab 已安装到
当前 Python 环境后才可用。

请只安装一个 backend extra 的 PyTorch：

```bash
# 仅 CPU 或非 NVIDIA 开发机器。
uv sync --extra torch-cpu

# NVIDIA 机器；选择驱动支持的 CUDA wheel。
uv sync --extra torch-cu118
uv sync --extra torch-cu121
uv sync --extra torch-cu124
uv sync --extra torch-cu126
uv sync --extra torch-cu128
uv sync --extra torch-cu130
```

如果不确定应使用哪个 CUDA build，请用 `nvidia-smi` 查看驱动，并参考 PyTorch 官方安装选择器。

创建 Genesis/Quadrants 和 Matplotlib 使用的项目本地缓存目录：

```bash
uv run genelab cache
```

## CLI

```bash
uv run genelab
uv run genelab --help
uv run genelab list robots
uv run genelab list envs
uv run genelab list tasks
```

## 核心 API

- `genelab.registry`：注册表、注册 helper 和扩展加载。
- `genelab.configs`：可复用 dataclass 配置，包括 `ManagerBasedEnvCfg` 和 `TaskCfg`。
- `genelab.lab`：注册表和 manager-based 环境原语的公共 API facade。
- `genelab.envs`、`genelab.robots`、`genelab.tasks`：注册 helper 的核心命名空间。
- `genelab.actuator`、`genelab.entity`、`genelab.scene`、`genelab.sensor`、
  `genelab.terrains` 和 `genelab.rl`：面向机器人研究代码的扩展命名空间。

下游项目应作为独立 Python 包存在，并通过 GeneLab 的 registry 和 extension hooks 注册机器人、
环境和任务。示例文档见 [examples/README.md](../examples/README.md)。
可用以下命令创建基础 external project 骨架：

```bash
uv run genelab project new my_robot_project
```

## 验证

```bash
uv run python -c "import genelab; print(genelab.__version__)"
uv run python -c "from genelab.lab import ManagerBasedEnvCfg; print(ManagerBasedEnvCfg.__name__)"
uv run python -c "import genesis; print(genesis.__version__)"
uv run pytest
uv run ruff check
uv run pyright
```

同步任意一个 `torch-*` extra 后，可验证当前 PyTorch build：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```
