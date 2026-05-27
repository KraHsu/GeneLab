<div align="center">

# 🧬 GeneLab

**面向 RL 与机器人研究的 Isaac Lab 风格 API —— 由 [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 提供仿真后端。**

熟悉的注册机器人 / 环境 / 任务、manager-based MDP 配置与 Typer CLI，
以 Genesis 作为轻量仿真后端 —— 无 USD/Kit，无厂商锁定。

[![CI](https://github.com/KraHsu/GeneLab/actions/workflows/ci.yml/badge.svg)](https://github.com/KraHsu/GeneLab/actions/workflows/ci.yml)
[![Docs](https://github.com/KraHsu/GeneLab/actions/workflows/docs.yml/badge.svg)](https://krahsu.github.io/GeneLab/zh/)
![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![Genesis](https://img.shields.io/badge/sim-Genesis-FF6F00)
![uv](https://img.shields.io/badge/deps-uv-DE5FE9)

[**English**](../README.md) · [**文档**](https://krahsu.github.io/GeneLab/zh/) · [**示例**](../examples/README.md)

</div>

---

## ✨ 特性

- **Isaac Lab 风格 API** —— 注册机器人 / 环境 / 任务，配合 manager-based 的
  actions、observations、rewards、events、terminations。
- **Genesis 后端** —— 快、轻量；无 USD/Kit，无 NVIDIA 锁定。
- **三个 RL 后端** —— `rsl_rl`、`skrl`、`stable_baselines3`，按 agent 配置类型自动分派。
- **开箱即用的 CLI** —— `train` / `play` / `eval` / `export` / `benchmark`，支持多 seed、多 GPU。
- **Asset zoo** —— Franka、Unitree G1 / Go1 / H1、ANYmal-C、UR10e、cartpole …… 按需下载。
- **可扩展** —— 下游项目通过干净的扩展 API 注册自己的机器人、环境与任务。

## 🚀 快速开始

```bash
uv sync --extra torch-cu128                 # 按你的 CUDA 选 torch extra（见下）
uv run genelab cache                        # 创建本地 仿真 / 绘图 缓存目录
uv run genelab list tasks                   # 看看注册了哪些任务
uv run genelab train GeneLab-Inverted-Pendulum-v0 --max_iterations 150
uv run genelab play  GeneLab-Inverted-Pendulum-v0 --vis
```

> `uv sync` 创建项目 venv 并安装 GeneLab + `uv.lock` 锁定的依赖。`uv run …` 在该 venv 内执行；
> 裸 `genelab` 命令只有在激活 `.venv` 后才可用。

## 📦 安装

环境要求：**Python ≥ 3.12** 与 [**uv**](https://docs.astral.sh/uv/)。

**只能选一个** `torch-*` extra —— 它们互斥：

| Extra | 硬件目标 |
|---|---|
| `torch-cpu`   | 纯 CPU 或非 NVIDIA 开发机 |
| `torch-cu126` | NVIDIA，CUDA 12.6 驱动 |
| `torch-cu128` | NVIDIA，CUDA 12.8 驱动 |
| `torch-cu130` | NVIDIA，CUDA 13.0 驱动 |

```bash
uv sync --extra torch-cpu        # 上面四选一
```

> **需要 PyTorch ≥ 2.8。** 旧版本 torch 会在 import 时报 `'torch<2.8.0' is not supported`，
> 且可能破坏 Genesis 运行假设。所有 `torch-*` extra 都 pin 了 `torch>=2.8.0`，`uv sync` 会自动拉
> 兼容的 wheel。PyTorch 只在 `cpu` / `cu126` / `cu128` / `cu130` 这几个 index 发布 2.8+ wheel
> （更旧的 `cu118` / `cu121` / `cu124` 有意不作为 extra 提供）。环境里已有旧 torch 时用
> `uv sync --reinstall-package torch --extra torch-cuXXX` 刷新。

## 🖥️ CLI

```bash
uv run genelab --help
uv run genelab list robots          # 已注册的机器人
uv run genelab list envs            # 已注册的环境
uv run genelab list tasks           # 已注册的任务
uv run genelab info  <task>         # 任务详情 + 可覆盖的配置路径
uv run genelab train <task> …       # 训练（后端由任务的 agent 配置决定）
uv run genelab play  <task> …       # rollout：--agent zero | random | trained
uv run genelab eval  <task> <ckpt>  # 确定性评估 → eval.json
uv run genelab export <task> <ckpt> # 导出策略 → TorchScript / ONNX
```

## 🧩 核心 API

- `genelab.registry` —— 注册表、注册辅助函数与扩展加载。
- `genelab.configs` —— 可复用的 dataclass 配置，包括 `ManagerBasedEnvCfg` 与 `TaskCfg`。
- `genelab.lab` —— 注册表与 manager-based 环境原语的公开 API 门面。
- `genelab.envs`、`genelab.robots`、`genelab.tasks` —— 注册表辅助的轻量核心命名空间。
- `genelab.actuator`、`genelab.entity`、`genelab.scene`、`genelab.sensor`、`genelab.terrains`、
  `genelab.rl` —— 机器人研究代码的扩展命名空间。
- `genelab.asset_zoo` —— 自带示例机器人（`g1`、`go1`、`anymal-c`、`franka`、`cartpole` ……）。
  通过 `ROBOTS` 注册表取用（`ROBOTS.get("g1")()`），或直接 import
  （`from genelab.asset_zoo import UnitreeG1Cfg`）。

下游项目放在各自的 Python 包里，通过 GeneLab 的注册表与扩展钩子注册机器人、环境和任务。
生成一个新脚手架：

```bash
uv run genelab project new my_robot_project
```

最小模板见 [`examples/external_project/`](../examples/external_project/README.md)。

## ✅ 验证

```bash
uv run python -c "import genelab, genesis; print(genelab.__version__, genesis.__version__)"
uv run python -c "from genelab.lab import ManagerBasedEnvCfg; print(ManagerBasedEnvCfg.__name__)"
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pyright
```

同步某个 `torch-*` extra 后，验证选中的 PyTorch 构建：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## 🛠️ 故障排查

<details>
<summary><b>GPU 空转 / 训练异常慢</b></summary>

<br>

`SimulationCfg.gpu` 默认是 **`False`（CPU 后端）**。CPU 后端下物理在 CPU 上步进、而 policy/张量
在 GPU 上 —— GPU 几乎空转，训练可能慢 **~50–100×**（接触多的任务如 Unitree G1 受影响最大）。
请在任务的 `SimulationCfg` 里设 `gpu=True`。若训练时 `nvidia-smi` 看到 GPU 占用接近 0%，基本就是
这个原因。更多见
[`docs/best-practices/reference-runs`](best-practices/reference-runs.zh.md)。

</details>

<details>
<summary><b>Hopper GPU（H100 / H200，SM 90）</b></summary>

<br>

Genesis 自带的 Quadrants 预编译 fatbin 在 `graph_do_while` 分派路径上不含 SM 90，所以任何任务在
H100 / H200 上构建场景时会报：

```
RuntimeError: Failed to load graph_do_while condition kernel fatbin (CUDA error 200).
This SM (90) may not be included in the fatbin
```

关闭 graph 分派：

```bash
export QD_GRAPH=0                     # 整个会话
QD_GRAPH=0 uv run genelab train …     # 或只针对单条命令
```

注意：`QD_GRAPH=0` 会关掉 CUDA-graph 批处理，明显拖慢**接触多**的仿真 ——
重的 locomotion 训练建议用非 Hopper 卡（Ada / Ampere）。

</details>

---

<div align="center">
<sub><a href="https://krahsu.github.io/GeneLab/zh/">文档</a> · <a href="../examples/README.md">示例</a> · 基于 <a href="https://github.com/Genesis-Embodied-AI/Genesis">Genesis</a></sub>
</div>
