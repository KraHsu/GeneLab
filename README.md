# GeneLab

GeneLab is an Isaac Lab-inspired API for RL and robotics research powered by
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis). It keeps the familiar shape of registered
robots, environments, tasks, manager-based MDP configuration, and CLI dispatch, while using Genesis
as the simulation backend.

## Goals

- Provide small registries for robots, environments, and tasks.
- Keep core API layers separate from example assets and demo scripts.
- Use manager-style config hooks for actions, observations, rewards, events, and terminations.
- Keep Genesis backend integration explicit and easy to extend.
- Support downstream robotics projects through a stable package layout and CLI.

## Requirements

- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/) for dependency management.

## Setup

Run setup from the repository root:

```bash
uv sync
uv run genelab --help
```

`uv sync` creates the project virtual environment, installs GeneLab from this checkout, and installs
the dependencies pinned by `uv.lock`. `uv run ...` runs commands inside that environment. A bare
`genelab` command works only after `.venv` is activated or GeneLab is installed into the active
Python environment.

If your workflow needs PyTorch directly, install exactly one backend extra:

```bash
# CPU-only or non-NVIDIA development machines.
uv sync --extra torch-cpu

# NVIDIA machines; choose the CUDA wheel supported by your driver.
uv sync --extra torch-cu118
uv sync --extra torch-cu121
uv sync --extra torch-cu124
uv sync --extra torch-cu126
uv sync --extra torch-cu128
uv sync --extra torch-cu130
```

If you are not sure which CUDA build to use, check `nvidia-smi` and follow the PyTorch installation
selector for your platform.

Create project-local cache folders used by Genesis/Quadrants and Matplotlib:

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

## Core API

- `genelab.registry`: registries, registration helpers, and extension loading.
- `genelab.configs`: reusable dataclass configs, including `ManagerBasedEnvCfg` and `TaskCfg`.
- `genelab.lab`: public API facade for registry and manager-based environment primitives.
- `genelab.envs`, `genelab.robots`, `genelab.tasks`: thin core namespaces for registry helpers.
- `genelab.actuator`, `genelab.entity`, `genelab.scene`, `genelab.sensor`, `genelab.terrains`,
  and `genelab.rl`: extension namespaces for robotics research code.

Downstream projects should live in their own Python packages and register robots, environments, and
tasks through GeneLab's registry and extension hooks. See
[`examples/external_project/`](examples/external_project/README.md) for a minimal package, or start
one with:

```bash
uv run genelab project new my_robot_project
```

## Documentation

- [Examples](examples/README.md): bundled tasks, demo scripts, config overrides, and downstream
  project integration.
- [中文 README](docs/README_CN.md): concise Chinese project overview.

## Verification

```bash
uv run python -c "import genelab; print(genelab.__version__)"
uv run python -c "from genelab.lab import ManagerBasedEnvCfg; print(ManagerBasedEnvCfg.__name__)"
uv run python -c "import genesis; print(genesis.__version__)"
uv run pytest
uv run ruff check
uv run pyright
```

After syncing one of the `torch-*` extras, verify the selected PyTorch build:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```
