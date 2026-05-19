# Installation

This page is the focused installation checklist. For the full learning path, start with the
[tutorial](../tutorial.md).

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.12 or newer |
| Dependency manager | `uv` |
| Simulator backend | `genesis-world>=0.4.7` |
| PyTorch | `torch>=2.8.0` through one `torch-*` extra |

## Sync the environment

From the repository root:

```bash
uv sync --extra torch-cpu
uv run genelab --version
```

Choose exactly one PyTorch extra:

| Extra | Target |
|---|---|
| `torch-cpu` | CPU-only or non-NVIDIA development machines. |
| `torch-cu126` | NVIDIA driver compatible with CUDA 12.6 wheels. |
| `torch-cu128` | NVIDIA driver compatible with CUDA 12.8 wheels. |
| `torch-cu130` | NVIDIA driver compatible with CUDA 13.0 wheels. |

Refresh an existing torch install:

```bash
uv sync --reinstall-package torch --extra torch-cu128
```

## Create local caches

```bash
uv run genelab cache
```

The command creates writable project-local cache directories and points `XDG_CACHE_HOME` and
`MPLCONFIGDIR` at `.cache/`.

## Verify imports

```bash
uv run python -c "import genelab; print(genelab.__version__)"
uv run python -c "from genelab.lab import TaskCfg; print(TaskCfg.__name__)"
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
uv run python -c "import genesis; print(genesis.__version__)"
```

## Install example extensions

The core package is a framework; tasks are registered by extensions.

```bash
uv pip install -e examples/inverted_pendulum
uv pip install -e examples/genelab_examples
uv run genelab list tasks
```

Install Unitree only when you want the larger humanoid examples:

```bash
uv pip install -e examples/unitree
```

## See also

- [Tutorial](../tutorial.md)
- [Debug Common Failures](../best-practices/debugging.md)
- [Build an Extension Project](../best-practices/extension-projects.md)
