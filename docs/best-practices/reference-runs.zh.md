# Reference Runs

本页是 GeneLab 自带任务的**复现地基**。按任务 × seed 列出收敛后的 return、
收敛步数、wall-clock 预算 — 用同一份 config 跑 `clone → train → eval`，应该
落到的数。

> **状态 — 2026-05-21**：五张表全部用 M1.7 reference batch 真实数据填好
> （8×H200、Genesis 0.4.7、`QD_GRAPH=0`）。G1 eval 必须 `QT_QPA_PLATFORM=
> offscreen` 才能跑（launcher 自己的 eval 一开始死在 Genesis init — cv2/Qt
> + tracking play_env 的 `episode_length_s = 1e9` 死循环，commit `a561ba0`
> 已 clamp 到 30 s 修掉）。Franka 那组按"已知异常"如实记录 —— 见 Franka 表
> 下的 **Franka 状态备注**。

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
#    --num-envs 必须和任务 train num_envs 一致（当前 64）；prefill loader
#    会断言 shape 对齐。
uv run python -m genelab_franka_pick_and_place.collect_demos \
    --num-envs 64 --steps 1000 \
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
| 1 | 39.944 | 0.026 | 150 | ~21 min | 10.3 s |
| 2 | 39.978 | 0.002 | 150 | ~20 min | 10.1 s |
| 3 | 39.991 | 0.001 | 150 | ~19 min | 10.1 s |

三个 seed 的 eval `length_mean = 1000.0`（episode 跑满时间上限不倒），策略
在预算上限解掉。`success_rate` 为 `null`（任务未 publish
`extras["is_success"]`）。

### `GeneLab-Double-Inverted-Pendulum-v0`

| Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 59.980 | 0.007 | 300 | ~85 min | 12.2 s |
| 2 | 59.986 | 0.003 | 300 | ~88 min | 14.2 s |
| 3 | 59.987 | 0.002 | 300 | ~85 min | 12.6 s |

三个 seed 的 eval `length_mean = 1200.0`。`success_rate` 为 `null`（同 IP）。

### `Genelab-Velocity-Flat-Unitree-G1-v0`

| Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 112.419 | 4.647 | 30 000 | ~18.7 h | 143.0 s |
| 2 | 93.417 | 3.921 | 30 000 | ~20.6 h | 161.0 s |
| 3 | 92.028 | 4.162 | 30 000 | ~19.8 h | 156.9 s |

三个 seed 的 eval `length_mean = 1000.0`（play_env `episode_length_s =
20 s` × 50 Hz）。`success_rate` 为 `null`。

### `Genelab-Tracking-Flat-Unitree-G1-v0`

| Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 137.800 | 0.005 | 30 000 | ~20.8 h | 212.8 s |
| 2 | 138.047 | 0.004 | 30 000 | ~20.6 h | 216.8 s |
| 3 | 138.122 | 0.007 | 30 000 | ~20.9 h | 216.0 s |

Eval `length_mean = 1500.0`。Tracking play_env 默认 `episode_length_s =
1e9` 是为 viewer 无限 playback 设的；`genelab eval` 把它 clamp 到 30 s（commit
`a561ba0`），所以 30 s × 50 Hz = 1500 步/ep，全部撞 cap 不 terminate。
三 seed 之间标准差非常紧 —— 收敛策略在 30 s 窗口内稳定跟随 motion clip。
`success_rate` 为 `null`。

### `GeneLab-Franka-Pick-And-Place-v0` (SAC+HER，demo 预填)

| Seed | 最终 `return_mean` | `return_std` | `success_rate` | 收敛 timestep | 训练 wall-clock |
|---|---|---|---|---|---|
| 1 | −95.996 | 19.595 | 0.11 | 1,843,200（预算上限）| 56 s |
| 2 | −97.998 | 14.000 | 0.04 | 1,843,200（预算上限）| 56 s |
| 3 | −92.199 | 26.519 | 0.11 | 1,843,200（预算上限）| 57 s |

三个 seed 的 eval `length_mean = 100.0`（固定 episode length）。

> **Franka 状态备注 —— 异常，未收敛**：
> 2 M-timestep 的预算在单卡 H200 上 **≈ 56 秒 wall-clock** 就被消耗掉，
> `model.learn()` 之后即返回；最终 success_rate 落在 **0.04–0.11**，基本
> 等于 demo 预填的 baseline。我们尚未根因定位的是 SB3 `HerReplayBuffer`
> auto-scaling、demo 预填、`learning_starts` 三者之间的交互 —— 梯度更新
> 循环在 `total_timesteps` 还远没真正用满有效探索时就退出了。
>
> 这几行**如实**记录，保证表格能从当前 `feat/m1-multi-seed-refs` 分支 +
> Genesis 0.4.7 + `QD_GRAPH=0` 直接复现。修这个 SAC/HER 交互**留给 M2**
> 处理；数字到 M2 阶段再回头修订。Franka 这行不要当作收敛 reference，
> 当作 **lower-bound 的 smoke baseline** 看就行。

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
