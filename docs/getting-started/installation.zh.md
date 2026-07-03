# 安装

专门的安装检查清单。完整学习路径请从[教程](../tutorial.md)开始。

## 要求

| 要求 | 版本 |
|---|---|
| Python | 3.12 或更新 |
| 依赖管理 | `uv` |
| 仿真后端 | `genesis-world>=1.2.0,<1.3.0`（Genesis 的次版本会改动 API，验证前先设上限） |
| PyTorch | 通过一个 `torch-*` extra 安装 `torch>=2.8.0` |

## 同步环境

在仓库根目录：

```bash
uv sync --extra torch-cpu
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

## 运行命令

后续文档统一以**裸 `genelab`**（以及裸 `python`）调用 CLI —— 而**不是** `uv run genelab`。

!!! warning "为什么不用 `uv run`？"

    `uv run` 会在**每条**命令前隐式执行一次 `uv sync`。而 PyTorch 各构建是互斥的 extra
    （`torch-cpu` / `torch-cu126` / `torch-cu128` / `torch-cu130`），**不在**默认同步集合里，
    因此每次 `uv run` 都会卸载并重装 torch、并改写上面选定的 extra —— 既慢，又可能悄悄把当前的
    CUDA 构建换掉。请改用下面两种方式之一。

=== "激活 venv（推荐）"

    每个 shell 激活一次，之后就能按文档里的写法运行裸命令：

    ```bash
    source .venv/bin/activate      # Windows：.venv\Scripts\activate
    genelab --version
    ```

=== "用 uv 但不重新同步"

    不想激活？给每条命令加 `--no-sync` 跳过隐式同步：

    ```bash
    uv run --no-sync genelab --version
    ```

    （文档里凡是 `genelab …` 或 `python …`，都可读作 `uv run --no-sync genelab …`。）

!!! note "为什么 `uv sync` 和 `uv pip install` 仍保留 `uv` 前缀"

    真正的坑只有 `uv run` —— 它在**执行**前会重新解析项目并把 venv 同步到该状态。而 `uv sync`
    是唯一一次有意指定 torch 构建的同步；`uv pip install -e …` 则是 `uv` 的
    pip 兼容安装器：它直接装进当前 venv，不会重新解析项目、也不碰 extra，因此绝不会重装 torch。
    两者都可以放心保留。

## 创建本地缓存

```bash
genelab cache
```

该命令创建项目本地可写缓存目录，并把 `XDG_CACHE_HOME` 和 `MPLCONFIGDIR` 指向 `.cache/`。

## 验证导入

```bash
python -c "import genelab; print(genelab.__version__)"
python -c "from genelab.lab import TaskCfg; print(TaskCfg.__name__)"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import genesis; print(genesis.__version__)"
```

## 安装示例扩展

核心包是框架；任务由扩展注册。

```bash
uv pip install -e examples/inverted_pendulum
uv pip install -e examples/genelab_examples
genelab list tasks
```

只有需要更大的人形机器人示例时再安装 Unitree：

```bash
uv pip install -e examples/unitree
```

## 另见

- [教程](../tutorial.md)
- [调试常见问题](../best-practices/debugging.md)
- [构建扩展项目](../best-practices/extension-projects.md)
