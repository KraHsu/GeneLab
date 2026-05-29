# Installation

A focused installation checklist. For the full learning path, start with the
[tutorial](../tutorial.md).

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.12 or newer |
| Dependency manager | `uv` |
| Simulator backend | `genesis-world>=0.4.7` |
| PyTorch | `torch>=2.8.0` through one `torch-*` extra |

## Syncing the environment

From the repository root:

```bash
uv sync --extra torch-cpu
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

## Running commands { #run-commands }

The rest of these docs invoke the CLI as a **bare `genelab`** (and a bare `python`) — *not*
`uv run genelab`.

!!! warning "Why not `uv run`?"

    `uv run` performs an implicit `uv sync` before **every** command. The PyTorch builds ship as
    mutually-exclusive extras (`torch-cpu` / `torch-cu126` / `torch-cu128` / `torch-cu130`) that are
    **not** part of the default sync set, so each `uv run` uninstalls and reinstalls torch and
    rewrites the extra selected above — slow, and it can silently flip the CUDA build. Use one
    of the two ways below instead.

=== "Activate the venv (recommended)"

    Activate once per shell, then run bare commands as shown throughout the docs:

    ```bash
    source .venv/bin/activate      # Windows: .venv\Scripts\activate
    genelab --version
    ```

=== "Use uv without re-syncing"

    Prefer not to activate? Skip the implicit sync on every call with `--no-sync`:

    ```bash
    uv run --no-sync genelab --version
    ```

    (Anywhere the docs write `genelab …` or `python …`, read it as `uv run --no-sync genelab …`.)

!!! note "Why `uv sync` and `uv pip install` keep the `uv` prefix"

    Only `uv run` is the footgun — it re-resolves the project and reconciles the venv before
    *executing*. `uv sync` is the one deliberate sync that selects the torch build, and
    `uv pip install -e …` is `uv`'s pip-compatible installer: it installs into the active venv
    without re-resolving the project or touching extras, so it never reinstalls torch. Both are
    safe to keep.

## Creating local caches

```bash
genelab cache
```

The command creates writable project-local cache directories and points `XDG_CACHE_HOME` and
`MPLCONFIGDIR` at `.cache/`.

## Verifying imports

```bash
python -c "import genelab; print(genelab.__version__)"
python -c "from genelab.lab import TaskCfg; print(TaskCfg.__name__)"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import genesis; print(genesis.__version__)"
```

## Installing example extensions

The core package is a framework; tasks are registered by extensions.

```bash
uv pip install -e examples/inverted_pendulum
uv pip install -e examples/genelab_examples
genelab list tasks
```

Install Unitree only for the larger humanoid examples:

```bash
uv pip install -e examples/unitree
```

## See also

- [Tutorial](../tutorial.md)
- [Debug Common Failures](../best-practices/debugging.md)
- [Build an Extension Project](../best-practices/extension-projects.md)
