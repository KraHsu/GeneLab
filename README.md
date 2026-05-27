<div align="center">

# 🧬 GeneLab

**An Isaac Lab–style API for RL & robotics research — powered by [Genesis](https://github.com/Genesis-Embodied-AI/Genesis).**

Familiar registries, manager-based MDP configs, and a Typer CLI, with Genesis as a
lightweight simulation backend — no USD/Kit, no vendor lock-in.

[![CI](https://github.com/KraHsu/GeneLab/actions/workflows/ci.yml/badge.svg)](https://github.com/KraHsu/GeneLab/actions/workflows/ci.yml)
[![Docs](https://github.com/KraHsu/GeneLab/actions/workflows/docs.yml/badge.svg)](https://krahsu.github.io/GeneLab/)
![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![Genesis](https://img.shields.io/badge/sim-Genesis-FF6F00)
![uv](https://img.shields.io/badge/deps-uv-DE5FE9)

[**中文**](docs/README_CN.md) · [**Documentation**](https://krahsu.github.io/GeneLab/) · [**Examples**](examples/README.md)

</div>

---

## ✨ Highlights

- **Isaac Lab–shaped API** — registered robots / environments / tasks, with manager-based
  actions, observations, rewards, events, and terminations.
- **Genesis backend** — fast and lightweight; no USD/Kit, no NVIDIA lock-in.
- **Three RL backends** — `rsl_rl`, `skrl`, and `stable_baselines3`, dispatched by agent-config type.
- **Batteries-included CLI** — `train` / `play` / `eval` / `export` / `benchmark`, with
  multi-seed and multi-GPU support.
- **Asset zoo** — Franka, Unitree G1 / Go1 / H1, ANYmal-C, UR10e, cartpole … fetched on demand.
- **Extensible** — downstream projects register their own robots, envs, and tasks through a
  clean extension API.

## 🚀 Quickstart

```bash
uv sync --extra torch-cu128                 # pick the torch extra for your CUDA (see below)
uv run genelab cache                        # create local sim / plot cache dirs
uv run genelab list tasks                   # see what's registered
uv run genelab train GeneLab-Inverted-Pendulum-v0 --max_iterations 150
uv run genelab play  GeneLab-Inverted-Pendulum-v0 --vis
```

> `uv sync` builds the project venv and installs GeneLab + the deps pinned by `uv.lock`.
> `uv run …` runs inside that venv; a bare `genelab` command works only after activating `.venv`.

## 📦 Installation

Requirements: **Python ≥ 3.12** and [**uv**](https://docs.astral.sh/uv/).

Pick **exactly one** `torch-*` extra — they are mutually exclusive:

| Extra | Hardware target |
|---|---|
| `torch-cpu`   | CPU-only or non-NVIDIA development machines |
| `torch-cu126` | NVIDIA, CUDA 12.6 driver |
| `torch-cu128` | NVIDIA, CUDA 12.8 driver |
| `torch-cu130` | NVIDIA, CUDA 13.0 driver |

```bash
uv sync --extra torch-cpu        # one of the above
```

> **PyTorch ≥ 2.8 required.** Genesis emits a `'torch<2.8.0' is not supported` warning on older
> builds and may break at runtime. All `torch-*` extras pin `torch>=2.8.0`, so `uv sync` pulls a
> compatible wheel automatically. PyTorch publishes 2.8+ wheels only on the `cpu` / `cu126` /
> `cu128` / `cu130` indices (older `cu118` / `cu121` / `cu124` are intentionally not offered).
> Refresh an older torch already in the env with
> `uv sync --reinstall-package torch --extra torch-cuXXX`.

## 🖥️ CLI

```bash
uv run genelab --help
uv run genelab list robots          # registered robots
uv run genelab list envs            # registered environments
uv run genelab list tasks           # registered tasks
uv run genelab info  <task>         # task detail + overridable cfg paths
uv run genelab train <task> …       # train (backend chosen by the task's agent cfg)
uv run genelab play  <task> …       # rollout: --agent zero | random | trained
uv run genelab eval  <task> <ckpt>  # deterministic eval → eval.json
uv run genelab export <task> <ckpt> # export policy → TorchScript / ONNX
```

## 🧩 Core API

- `genelab.registry` — registries, registration helpers, and extension loading.
- `genelab.configs` — reusable dataclass configs, including `ManagerBasedEnvCfg` and `TaskCfg`.
- `genelab.lab` — public API facade for registry and manager-based environment primitives.
- `genelab.envs`, `genelab.robots`, `genelab.tasks` — thin core namespaces for registry helpers.
- `genelab.actuator`, `genelab.entity`, `genelab.scene`, `genelab.sensor`, `genelab.terrains`,
  `genelab.rl` — extension namespaces for robotics research code.
- `genelab.asset_zoo` — bundled example robots (`g1`, `go1`, `anymal-c`, `franka`, `cartpole`, …).
  Fetch via the `ROBOTS` registry (`ROBOTS.get("g1")()`) or import directly
  (`from genelab.asset_zoo import UnitreeG1Cfg`).

Downstream projects live in their own Python packages and register robots, environments, and
tasks through GeneLab's registry and extension hooks. Scaffold a fresh one:

```bash
uv run genelab project new my_robot_project
```

The minimal template lives at [`examples/external_project/`](examples/external_project/README.md).

## ✅ Verification

```bash
uv run python -c "import genelab, genesis; print(genelab.__version__, genesis.__version__)"
uv run python -c "from genelab.lab import ManagerBasedEnvCfg; print(ManagerBasedEnvCfg.__name__)"
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pyright
```

After syncing a `torch-*` extra, verify the selected PyTorch build:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## 🛠️ Troubleshooting

<details>
<summary><b>GPU sits idle / training is unexpectedly slow</b></summary>

<br>

`SimulationCfg.gpu` defaults to **`False` (CPU backend)**. With the CPU backend the physics
steps on the CPU while the policy/tensors sit on the GPU — the GPU stays near-idle and training
can be **~50–100× slower** (contact-heavy tasks like Unitree G1 are hit hardest). Set `gpu=True`
in your task's `SimulationCfg`. If `nvidia-smi` shows your training GPU near 0 % during steps,
this is almost always why. See
[`docs/best-practices/reference-runs`](docs/best-practices/reference-runs.en.md) for details.

</details>

<details>
<summary><b>Hopper GPUs (H100 / H200, SM 90)</b></summary>

<br>

Genesis ships precompiled Quadrants kernel fatbins that omit SM 90 for the `graph_do_while`
dispatch path, so any task aborts during scene build with:

```
RuntimeError: Failed to load graph_do_while condition kernel fatbin (CUDA error 200).
This SM (90) may not be included in the fatbin
```

Disable graph dispatch:

```bash
export QD_GRAPH=0                     # for the session
QD_GRAPH=0 uv run genelab train …     # or for a single command
```

Note: `QD_GRAPH=0` disables CUDA-graph batching and noticeably slows **contact-heavy** sims —
prefer a non-Hopper GPU (Ada / Ampere) for heavy locomotion training.

</details>

---

<div align="center">
<sub><a href="https://krahsu.github.io/GeneLab/">Documentation</a> · <a href="examples/README.md">Examples</a> · built on <a href="https://github.com/Genesis-Embodied-AI/Genesis">Genesis</a></sub>
</div>
