# 安装

GeneLab 需要 **Python 3.12 或更新版本**，使用 [`uv`](https://docs.astral.sh/uv/) 管理依赖。

## 1. 同步环境

在仓库根目录运行：

```bash
uv sync
uv run genelab --help
```

`uv sync` 会创建项目虚拟环境，从当前 checkout 安装 GeneLab，并安装 `uv.lock` 锁定的依赖。
`uv run ...` 在该环境中运行命令。裸 `genelab` 命令只有在 `.venv` 已激活、或 GeneLab 已安装
到当前 Python 环境后才可用。

## 2. 挑选一个 PyTorch extra

`torch-*` extras **互斥**：

| Extra | 目标硬件 |
|-------|---------|
| `torch-cpu` | 仅 CPU 或非 NVIDIA 开发机器。 |
| `torch-cu126` | NVIDIA，CUDA 12.6 驱动。 |
| `torch-cu128` | NVIDIA，CUDA 12.8 驱动。 |
| `torch-cu130` | NVIDIA，CUDA 13.0 驱动。 |

```bash
uv sync --extra torch-cpu        # 从上表挑一个
```

!!! warning "PyTorch 版本要求"
    Genesis 要求 `torch>=2.8.0`，旧版本在导入时会报 `'torch<2.8.0' is not supported`，并可能
    破坏 Genesis 的运行时假设。所有 `torch-*` extras 都固定 `torch>=2.8.0`，`uv sync` 会自动
    拉取兼容 wheel。PyTorch 只在 `cpu`、`cu126`、`cu128`、`cu130` 索引提供 2.8+ wheel，因此
    较旧的 CUDA flavour（`cu118` / `cu121` / `cu124`）不再以 extra 形式提供。环境里已存在
    旧 torch 时，可用 `uv sync --reinstall-package torch --extra torch-cuXXX` 刷新。

不确定使用哪个 CUDA build 时，用 `nvidia-smi` 查看驱动版本。

## 3. 初始化项目本地缓存

Genesis、Quadrants、Matplotlib 都需要可写缓存目录。CLI 一条命令完成目录与环境变量配置：

```bash
uv run genelab cache
```

它会把 `XDG_CACHE_HOME` 和 `MPLCONFIGDIR` 指向 `.cache/`，避免仿真器写到用户主目录。

## 4. 验证

```bash
uv run python -c "import genelab; print(genelab.__version__)"
uv run python -c "from genelab.lab import ManagerBasedEnvCfg; print(ManagerBasedEnvCfg.__name__)"
uv run python -c "import genesis; print(genesis.__version__)"
uv run pytest
uv run ruff check
uv run pyright
```

验证当前 PyTorch build：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## See also

- [快速开始](quickstart.md)
