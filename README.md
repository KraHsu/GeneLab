# GeneLab

[中文](docs/README_CN.md)|EN

GeneLab is an Isaac Lab-inspired API for RL and robotics research powered by
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis). It keeps the familiar shape of registered
robots, environments, tasks, manager-based MDP configuration, and CLI dispatch, while using Genesis
as the simulation backend.

- [Documentation](https://krahsu.github.io/GeneLab/) — full guides, CLI reference, concepts,
  and auto-generated API docs. Bilingual (English default, 中文 at `/zh/`).
- [Examples](examples/README.md) — bundled tasks, demo scripts, config overrides, and downstream
  project integration.

## Goals

- Small registries for robots, environments, and tasks.
- Core API layers separated from example assets and demo scripts.
- Manager-style config hooks for actions, observations, rewards, events, and terminations.
- Explicit, easy-to-extend Genesis backend integration.
- A stable package layout and CLI for downstream robotics projects.

## Requirements

- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/) for dependency management.

## Setup

From the repository root:

```bash
uv sync
uv run genelab --help
```

`uv sync` creates the project virtual environment, installs GeneLab from this checkout, and installs
the dependencies pinned by `uv.lock`. `uv run ...` runs commands inside that environment. A bare
`genelab` command works only after `.venv` is activated or GeneLab is installed into the active
Python environment.

Pick exactly one `torch-*` extra — they are **mutually exclusive**:

| Extra | Hardware target |
|-------|----------------|
| `torch-cpu` | CPU-only or non-NVIDIA development machines. |
| `torch-cu126` | NVIDIA, CUDA 12.6 driver. |
| `torch-cu128` | NVIDIA, CUDA 12.8 driver. |
| `torch-cu130` | NVIDIA, CUDA 13.0 driver. |

```bash
uv sync --extra torch-cpu        # one of the above
```

> **PyTorch version requirement.** Genesis requires `torch>=2.8.0` — older builds emit a
> `'torch<2.8.0' is not supported` warning at import time and may break Genesis runtime
> assumptions. All `torch-*` extras pin `torch>=2.8.0`, so `uv sync` will pull a compatible
> wheel automatically. PyTorch only publishes 2.8+ wheels on the `cpu`, `cu126`, `cu128`, and
> `cu130` indices; older CUDA flavours (`cu118` / `cu121` / `cu124`) are intentionally not
> offered as extras. An older `torch` already in the environment can be refreshed with
> `uv sync --reinstall-package torch --extra torch-cuXXX`.

Create project-local cache folders used by Genesis/Quadrants and Matplotlib:

```bash
uv run genelab cache
```

## CLI

```bash
uv run genelab --help
uv run genelab list robots
uv run genelab list envs
uv run genelab list tasks
```

## Core API

- `genelab.registry` — registries, registration helpers, and extension loading.
- `genelab.configs` — reusable dataclass configs, including `ManagerBasedEnvCfg` and `TaskCfg`.
- `genelab.lab` — public API facade for registry and manager-based environment primitives.
- `genelab.envs`, `genelab.robots`, `genelab.tasks` — thin core namespaces for registry helpers.
- `genelab.actuator`, `genelab.entity`, `genelab.scene`, `genelab.sensor`, `genelab.terrains`,
  and `genelab.rl` — extension namespaces for robotics research code.
- `genelab.asset_zoo` — bundled example robots (`g1`, `go1`, `anymal-c`, `franka`, `cartpole`).
  Not part of the core facade: fetch via the `ROBOTS` registry (`ROBOTS.get("g1")()`) or import
  the config directly (`from genelab.asset_zoo import UnitreeG1Cfg`).

Downstream projects live in their own Python packages and register robots, environments, and
tasks through GeneLab's registry and extension hooks. The minimal template lives at
[`examples/external_project/`](examples/external_project/README.md); a fresh scaffold is
generated with:

```bash
uv run genelab project new my_robot_project
```

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

## Troubleshooting

### Hopper GPUs (H100 / H200, SM 90)

Genesis ships precompiled Quadrants kernel fatbins that do not include SM 90 for
the `graph_do_while` graph dispatch path. Launching any task on an H100 or H200
aborts during scene build with:

```
RuntimeError: Failed to load graph_do_while condition kernel fatbin (CUDA error 200).
This SM (90) may not be included in the fatbin
```

Disable the graph dispatch with `QD_GRAPH=0`:

```bash
QD_GRAPH=0 uv run genelab train ...
```

Or export it for the session:

```bash
export QD_GRAPH=0
```
