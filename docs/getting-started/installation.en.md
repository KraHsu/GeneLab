# Installation

GeneLab requires **Python 3.12 or newer** and uses [`uv`](https://docs.astral.sh/uv/) for
dependency management.

## 1. Sync the environment

From the repository root:

```bash
uv sync
uv run genelab --help
```

`uv sync` creates the project virtual environment, installs GeneLab from this checkout, and
installs the dependencies pinned by `uv.lock`. `uv run ...` runs commands inside that
environment. A bare `genelab` command works only after `.venv` is activated or GeneLab is
installed into the active Python environment.

## 2. Pick exactly one PyTorch extra

The `torch-*` extras are **mutually exclusive**:

| Extra | Hardware target |
|-------|----------------|
| `torch-cpu` | CPU-only or non-NVIDIA development machines. |
| `torch-cu126` | NVIDIA, CUDA 12.6 driver. |
| `torch-cu128` | NVIDIA, CUDA 12.8 driver. |
| `torch-cu130` | NVIDIA, CUDA 13.0 driver. |

```bash
uv sync --extra torch-cpu        # one of the above
```

!!! warning "PyTorch version requirement"
    Genesis requires `torch>=2.8.0` — older builds emit a `'torch<2.8.0' is not supported`
    warning at import time and may break Genesis runtime assumptions. All `torch-*` extras pin
    `torch>=2.8.0`, so `uv sync` will pull a compatible wheel automatically. PyTorch only
    publishes 2.8+ wheels on the `cpu`, `cu126`, `cu128`, and `cu130` indices; older CUDA
    flavours (`cu118` / `cu121` / `cu124`) are intentionally not offered as extras. An older
    `torch` already in the environment can be refreshed with
    `uv sync --reinstall-package torch --extra torch-cuXXX`.

Run `nvidia-smi` to confirm the driver version when unsure which CUDA build to use.

## 3. Initialize project-local caches

Genesis, Quadrants, and Matplotlib all require a writable cache directory. The CLI sets up both
the directory layout and the matching environment variables:

```bash
uv run genelab cache
```

This sets `XDG_CACHE_HOME` and `MPLCONFIGDIR` under `.cache/`, so the simulator never writes to
the user's home directory.

## 4. Verify

```bash
uv run python -c "import genelab; print(genelab.__version__)"
uv run python -c "from genelab.lab import ManagerBasedEnvCfg; print(ManagerBasedEnvCfg.__name__)"
uv run python -c "import genesis; print(genesis.__version__)"
uv run pytest
uv run ruff check
uv run pyright
```

Verify the selected PyTorch build:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## See also

- [Quickstart](quickstart.md)
