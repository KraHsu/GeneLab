# 安装

GeneLab 需要 **Python 3.12 或更新版本**，使用 [`uv`](https://docs.astral.sh/uv/) 管理依赖。

## 1. 同步环境

在仓库根目录运行：

```bash
uv sync
uv run genelab --help
```

`uv sync` 会创建项目虚拟环境，从当前 checkout 安装 GeneLab，并安装 `uv.lock` 锁定的依赖。
`uv run ...` 会在该环境中运行命令。裸 `genelab` 命令只有在 `.venv` 已激活、或 GeneLab 已安装
到当前 Python 环境后才可用。

## 2. 挑选一个 PyTorch extra（互斥）

`torch-*` extras **互斥**，只能挑选一个匹配你的硬件：

```bash
# 仅 CPU 或非 NVIDIA 开发机器。
uv sync --extra torch-cpu

# NVIDIA 机器，按驱动支持的 CUDA wheel 挑选。
uv sync --extra torch-cu126
uv sync --extra torch-cu128
uv sync --extra torch-cu130
```

!!! warning "PyTorch 版本要求"
    Genesis 要求 `torch>=2.8.0`，旧版本在导入时会报 `'torch<2.8.0' is not supported`，并可能
    破坏 Genesis 的运行时假设。所有 `torch-*` extras 都固定 `torch>=2.8.0`，`uv sync` 会自动
    拉取兼容 wheel。PyTorch 只在 `cpu`、`cu126`、`cu128`、`cu130` 索引提供 2.8+ wheel，因此
    较旧的 CUDA flavour（`cu118` / `cu121` / `cu124`）不再以 extra 形式提供。如果环境里已经
    存在旧 torch，可用 `uv sync --reinstall-package torch --extra torch-cuXXX` 刷新。

不确定使用哪个 CUDA build 时，用 `nvidia-smi` 查看驱动，并参考 PyTorch 官方安装选择器。

## 3. 初始化项目本地缓存

Genesis、Quadrants、Matplotlib 都需要可写缓存目录。一条命令同时创建项目本地目录和环境变量：

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

同步任意 `torch-*` extra 后，可验证当前 PyTorch build：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

下一步：阅读 [快速开始](quickstart.md) 运行一个已注册任务。
