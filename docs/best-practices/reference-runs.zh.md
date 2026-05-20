# Reference Runs

本页是 GeneLab 自带任务的**复现地基**。按任务 × seed 列出收敛后的 return、
收敛步数、wall-clock 预算 — 用同一份 config 跑 `clone → train → eval`，应该
落到的数。

> **状态 — 2026-05-20**：**复现协议已定稿**；**reference 数字 TBD**，跟在
> ROADMAP M1.7 下。下表当作 schema 看；等真正的 run 在一个固定 Genesis pin
> 上跑完后再单独提 PR 填进来。你自己跑过的话，把数字 + 曲线挂到 M1.7 跟踪
> issue 上即可。

## Reference 任务

这五个任务覆盖 GeneLab 内置的 locomotion + manipulation 两条线：

| Task ID | Backend (默认 agent) | 预算 | 备注 |
|---|---|---|---|
| `GeneLab-Inverted-Pendulum-v0` | rsl_rl PPO | 150 iter × 4096 envs | 小 cartpole；当 smoke target 用。 |
| `GeneLab-Double-Inverted-Pendulum-v0` | rsl_rl PPO | 300 iter × 4096 envs | 难一些的 cartpole。 |
| `Genelab-Velocity-Flat-Unitree-G1-v0` | rsl_rl PPO | 30k iter × 4096 envs | Unitree G1 平地速度跟踪。 |
| `Genelab-Tracking-Flat-Unitree-G1-v0` | rsl_rl PPO | 30k iter × 4096 envs | Unitree G1 平地动作跟踪。 |
| `GeneLab-Franka-Pick-And-Place-v0` | sb3 SAC + HER | 2M timesteps × 2048 envs | 目标条件抓取；需要离线 demo 预填（见下方协议）。 |

> **注**：dev 把 Franka example 收敛到唯一 SAC+HER 配置。早先的
> `Cartesian-v0` / `skrl-v0` / `sb3-v0` / `sb3-her-v0` 变体已经不再注册，
> 故意不收在这套里。

## 复现协议

### 通用路径（4 / 5 任务）

Cartpole + G1 都是 rsl_rl PPO，复现就用 multi-seed CLI 直接来：

```bash
# 1. 三个 seed 训练（cartpole 大小可以 parallel=3；G1 单卡用 parallel=1 防 OOM）。
genelab train <TASK> \
    --seeds 1,2,3 --parallel <P> \
    --log_dir logs/reference/<TASK>/<DATE>

# 2. 对每个 seed 的最终 checkpoint 做 deterministic eval。
for s in 1 2 3; do
  genelab eval <TASK> \
    "logs/reference/<TASK>/<DATE>/seed_${s}/model_final.pt" \
    --num-envs 64 --episodes 100 --seed 0 \
    --out "logs/reference/<TASK>/<DATE>/seed_${s}/eval.json"
done
```

下表数字读 `eval.json`。

### Franka SAC+HER 路径

`GeneLab-Franka-Pick-And-Place-v0` 是 goal-conditioned SAC+HER，训练前必须
离线 demo 预填，否则 cold-start replay buffer 永远见不到成功轨迹：

```bash
# 1. 用脚本化 FSM 收 demo（一次性，与 seed 无关）。
uv run python -m genelab_franka_pick_and_place.collect_demos \
    --num-envs 32 --steps 6400 \
    --out logs/reference/franka-pp/demos.npz

# 2. 三个 seed 训练 — 每个子进程通过 GENELAB_SB3_DEMO_PATH 读 demo 文件
#    （或在 cfg 里设 agent.demo_path）。
GENELAB_SB3_DEMO_PATH=logs/reference/franka-pp/demos.npz \
  genelab train GeneLab-Franka-Pick-And-Place-v0 \
    --seeds 1,2,3 --parallel 1 \
    --log_dir logs/reference/franka-pp/<DATE>

# 3. 对每个 seed 保存的 model.zip（SB3 原生格式）做 eval。
for s in 1 2 3; do
  genelab eval GeneLab-Franka-Pick-And-Place-v0 \
    "logs/reference/franka-pp/<DATE>/seed_${s}/model.zip" \
    --num-envs 64 --episodes 100 --seed 0 \
    --out "logs/reference/franka-pp/<DATE>/seed_${s}/eval.json"
done
```

Franka 任务**目前无法**通过 `genelab export` 导出 —— HER 的 `Dict` 观测在
M1.3 限制清单里。去掉这个限制是后续 M 系列 follow-up。

### 硬件

训练用一块 CUDA GPU（≥ 12 GB VRAM）。Deterministic eval 用 CPU 能跑但比
GPU vectorized 慢很多。

## Reference 数字

### `GeneLab-Inverted-Pendulum-v0`

| Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |

### `GeneLab-Double-Inverted-Pendulum-v0`

| Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |

### `Genelab-Velocity-Flat-Unitree-G1-v0`

| Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |

### `Genelab-Tracking-Flat-Unitree-G1-v0`

| Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |

### `GeneLab-Franka-Pick-And-Place-v0` (SAC+HER，demo 预填)

| Seed | 最终 `return_mean` | `return_std` | `success_rate` | 收敛 timestep | 训练 wall-clock |
|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |

## 训练曲线

曲线在 reference run 跑完后从 TensorBoard 导出。预期布局：

```
logs/reference/<TASK>/<DATE>/seed_<S>/
├── events.out.tfevents.*   # TensorBoard
├── ckpts/                  # checkpoints（或 model_<N>.pt 直接在该目录
│                           #   下，看 backend）
├── eval.json               # `genelab eval` 写出
└── (可选) curves.png       # 本文档引用的截图
```

run 跑完之前这一节有意留空 —— 上面的 schema 就是 PR 填进来时要对齐的格式。

## 方法学注意

- **Seeds 1、2、3 是 GeneLab 的标准三元组。** 本文档里某个任务用了别的 seed
  必须解释为什么（例如某 seed 在该任务上踩到 Genesis init 的 degenerate
  情况）。
- **Eval seed 固定为 0。** 这样不同 seed 之间、以及对同一 seed 重新跑此协议
  时，eval 的 rollout 轨迹完全一致 —— `return_mean` 上的方差只反映训练方差。
- **Locomotion 任务没有 `success_rate`**。本 revision 的 locomotion 任务没
  publish `extras["is_success"]`，doc 报 `null` 而不是凭空编一个阈值。
  Manipulation 任务（Franka）在 goal-reach termination 里 emit `is_success`，
  所以那边字段是填的。
- **Genesis 版本 pin。** 跑出这些数字时用的版本记在每个 `eval.json` 顶部
  （`evaluated_at` 字段 + 同目录下 `params/env.json` snapshot）。换 Genesis
  版本重跑**不保证**数字完全复现。

## 本文档不是

- **不是** benchmark suite —— 那是 M3.8（`genelab benchmark`）。
- **不是** 排行榜。这里的数字是 GeneLab 自己的复现 sanity check；社区数据
  走 benchmark suite。
- **不是** 调参指南。看 `best-practices/rl-experiments` 了解 curriculum / DR
  / reward weight 的选择 —— 那些是这些数字的上游。
