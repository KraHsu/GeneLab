# 参考指标

GeneLab 自带任务的**复现地基**。按任务 × seed 列出收敛后的 return、
收敛步数、wall-clock 预算 — 用同一份 config 跑 `clone → train → eval`，应该
落到的数。

## 参考任务

这六个任务覆盖 GeneLab 内置的 locomotion + manipulation 两条线：

| Task ID | Backend (默认 agent) | 预算 | 备注 |
|---|---|---|---|
| `GeneLab-Inverted-Pendulum-v0` | rsl_rl PPO | 150 iter × 4096 envs | 小 cartpole；当 smoke target 用。 |
| `GeneLab-Double-Inverted-Pendulum-v0` | rsl_rl PPO | 300 iter × 4096 envs | 难一些的 cartpole。 |
| `Genelab-Velocity-Flat-Unitree-G1-v0` | rsl_rl PPO | 30k iter × 4096 envs | Unitree G1 平地速度跟踪。 |
| `Genelab-Velocity-Rough-Unitree-G1-v0` | rsl_rl PPO | 6k iter × 4096 envs | Unitree G1 在 10 级混合地形课程上的速度跟踪。 |
| `Genelab-Tracking-Flat-Unitree-G1-v0` | rsl_rl PPO | 30k iter × 4096 envs | Unitree G1 平地动作跟踪。 |
| `Genelab-Velocity-Flat-Unitree-Go1-v0` | rsl_rl PPO | 3k iter × 4096 envs | Unitree Go1 四足平地速度跟踪；可部署的纯本体感知 actor（无基座线速度传感器）。 |
| `Genelab-Velocity-Rough-Unitree-Go1-v0` | rsl_rl PPO | (WIP) | Unitree Go1 在 10 级混合地形课程上。**暂未作为参考** —— 从零 3k 迭代陷入原地站立局部最优（仅约 1–2% 指令速度）；需要更大预算（约 6k）+ 更易的课程起步。 |
| `Genelab-Velocity-Flat-Unitree-Go2W-v0` | rsl_rl PPO | 6k+6k+4k iter × 4096 envs（两阶段） | Unitree Go2-W **轮式**四足,**轮腿混合**全向速度跟踪:锁轮 crab-walk 第一阶段 → 解锁轮第二阶段(镜像对称增强 + L1 跟踪项)。sim2sim 加固(5 帧堆叠、startup 域随机化、动作噪声与延迟)。 |
| `GeneLab-Franka-Pick-And-Place-v0` | sb3 SAC + HER | 2M timesteps × 64 envs | 目标条件抓取；需要离线 demo 预填（见下方协议）。 |

## 复现协议

### 通用路径（5 / 6 任务）

Cartpole + G1 都是 rsl_rl PPO，复现就用 multi-seed CLI：

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
python -m genelab_franka.collect_demos \
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

Franka 任务**目前无法**通过 `genelab export` 导出。导出目前只支持 flat-tensor
观测，而 SAC+HER 使用 goal-conditioned `Dict` 观测。

### 硬件

训练用一块 CUDA GPU（≥ 12 GB VRAM）。Deterministic eval 用 CPU 能跑但比
GPU vectorized 慢很多。

!!! warning "仿真必须跑在 GPU 后端"
    `SimulationCfg.gpu` 默认是 **`False`（CPU 后端）**。CPU 后端下物理在 CPU 上步进、而
    policy/张量在 GPU 上，导致 **GPU 全程空闲、训练慢 ~50–100×**（G1 这类接触多的任务从
    几秒/迭代变成几百秒/迭代）。自带的可训练任务都设了 `gpu=True`；**自定义任务也必须设**。
    若训练时 `nvidia-smi` 看到 GPU 占用接近 0%，基本就是这个原因。

!!! note "Hopper（H100/H200）与多卡注意事项"
    - **Hopper（SM 90）** 上必须设 `QD_GRAPH=0`（Genesis 没有 SM 90 的 `graph_do_while`
      fatbin）；这会关掉 CUDA-graph 批处理，严重拖慢**接触多**的仿真。复现 locomotion 建议用
      非 Hopper 卡（Ada / Ampere）。
    - 多卡（`genelab train --gpus N`）对 **G1 几乎不提速**（每步固定开销 + PCIe all-reduce
      主导）。多 seed 复现请**一卡一个 seed**，而不是一个 seed 摊到多卡。
    - 4096 envs 的 RL 训练基本是 **CPU 受限**、要吃满整机；同机并发多个这种训练会过订阅 CPU、
      超线性变慢。墙钟会变差，但奖励是确定性的，所以复现数值不受并发影响。

## 参考数字

下表为 Genesis **v1.0** 的数字。v0.4.7 的旧数字保留在每个任务旁边的
"Previous: v0.4.7" admonition 里，方便 re-baseline 对比。

!!! note "v1.0 数字所用硬件"

    - **Cartpole IP / DIP、Franka** — NVIDIA H200（141 GB HBM3），驱动
      570.211.01，CUDA 12.8，`QD_GRAPH=0`（Hopper SM 90 没有
      `graph_do_while` fatbin，必须设）。Cartpoles 用 `--parallel 3` 在
      单张 GPU 上并行；Franka 用一卡一 seed 跑在 GPU 0–2 上。
    - **G1-Velocity-Flat、G1-Tracking-Flat** — 4× NVIDIA GeForce RTX 4090
      （每张 24 GB），驱动 580.159.03，CUDA 12.8。G1-Velocity 用一卡一
      seed 跑在 GPU 0–2 上；G1-Tracking 的 seed 1 在 Phase A 与
      Velocity 并行跑在 GPU 3 上，seed 2/3 在 Velocity 跑完后才在 GPU
      0/1 上跑（Phase B）。跨任务在同一台机上并发触发了文档自身警告的
      CPU 过订阅 —— reward 数字是确定性的不受影响，但下面的训练墙钟是
      **并发跑的墙钟**，不是单 job 数字（同卡单 job 跑按 `train.log`
      里的 `Time elapsed` 计数约快 3–6×）。
    - **Go1-Velocity-Flat**、**Go2W-Velocity-Flat** —— 单张 NVIDIA
      GeForce RTX 5060 Ti（16 GB），本地工作站。都是**单 seed**（seed 42,
      任务 cfg 默认）跑,而不是 G1 那种三 seed 集群 sweep —— 算 smoke 级
      参考,不提供跨 seed 方差。

### `GeneLab-Inverted-Pendulum-v0`

| Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 39.977 | 0.002 | 150 | ~2.5 min | 12.6 s |
| 2 | 39.994 | 0.001 | 150 | ~2.5 min | 12.4 s |
| 3 | 39.986 | 0.001 | 150 | ~2.5 min | 12.6 s |

三个 seed 的 eval `length_mean = 1000.0`（episode 跑满时间上限不倒），策略
在预算上限解掉。`success_rate` 为 `null`（任务未 publish
`extras["is_success"]`）。

!!! note "Previous: v0.4.7"

    | Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
    |---|---|---|---|---|---|
    | 1 | 39.944 | 0.026 | 150 | ~21 min | 10.3 s |
    | 2 | 39.978 | 0.002 | 150 | ~20 min | 10.1 s |
    | 3 | 39.991 | 0.001 | 150 | ~19 min | 10.1 s |

### `GeneLab-Double-Inverted-Pendulum-v0`

| Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 59.933 | 0.020 | 300 | ~3.5 min | 16.3 s |
| 2 | 59.914 | 0.174 | 300 | ~3.5 min | 16.4 s |
| 3 | 59.897 | 0.023 | 300 | ~3.5 min | 16.4 s |

三个 seed 的 eval `length_mean = 1200.0`。`success_rate` 为 `null`（同 IP）。

!!! note "Previous: v0.4.7"

    | Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
    |---|---|---|---|---|---|
    | 1 | 59.980 | 0.007 | 300 | ~85 min | 12.2 s |
    | 2 | 59.986 | 0.003 | 300 | ~88 min | 14.2 s |
    | 3 | 59.987 | 0.002 | 300 | ~85 min | 12.6 s |

### `Genelab-Velocity-Flat-Unitree-G1-v0`

actor observation **不含 `base_lin_vel`**（与 rough 同一改动：从 actor 移除、仅 critic 保留；见
[sim2real](sim2real.md)）。下表为该配置（commit `d653aa9`）的数字。

| Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 112.02 | 4.85 | 30 000 | ~28 h | 165.2 s |
| 2 | 91.996 | 3.82 | 30 000 | ~28 h | 162.1 s |
| 3 | 111.85 | 5.10 | 30 000 | ~28 h | 163.6 s |

三个 seed 的 eval `length_mean = 1000.0`（play_env `episode_length_s = 20 s` × 50 Hz，策略全程
不摔）。`success_rate` 为 `null`。seed 1/3 与带 `base_lin_vel` 的 v1.0 baseline 持平（112.04 /
113.16）；**seed 2 落在 92（低 return 但稳定:length 1000、std 3.8）** —— 属 flat 任务已知的 seed
方差（v0.4.7 baseline 就有 92–93 的 seed），非删 `base_lin_vel` 的系统性损失。本 sweep 在 4× RTX
4090 上与 rough 并发、错峰跑,训练 wall-clock 为并发墙钟（~28 h 量级,非独占单 job）。

!!! warning "Eval 需要 `auto_reset` 修复（commit `d56158c`）"
    `genelab eval` 用 play_env（`auto_reset=False`）构建,evaluator 依赖 env 自动复位收集 episode；
    未打此 fix 时 eval 会坍缩成垃圾（退化的 ~1 步 episode）。上表数字均为修复后所得。

!!! note "Previous：带 `base_lin_vel`（v1.0 baseline）"

    actor 含 `base_lin_vel`。

    | Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
    |---|---|---|---|---|---|
    | 1 | 112.038 | 4.816 | 30 000 | ~28 h | 163.5 s |
    | 2 | 112.871 | 4.918 | 30 000 | ~28 h | 160.3 s |
    | 3 | 113.163 | 4.850 | 30 000 | ~28 h | 160.9 s |

!!! note "Previous: v0.4.7"

    | Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
    |---|---|---|---|---|---|
    | 1 | 112.419 | 4.647 | 30 000 | ~18.7 h | 143.0 s |
    | 2 | 93.417 | 3.921 | 30 000 | ~20.6 h | 161.0 s |
    | 3 | 92.028 | 4.162 | 30 000 | ~19.8 h | 156.9 s |

### `Genelab-Velocity-Rough-Unitree-G1-v0`

本任务的 actor observation **不含 `base_lin_vel`**（从 actor 移除、仅 critic 作为特权信号保留；
真实 G1 无直接基座线速度传感器，见 [sim2real](sim2real.md)）。下表为该配置（commit `d653aa9`）的数字。

| Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 82.96 | 29.20 | 6 000 | ~6.9 h | 516 s* |
| 2 | 85.51 | 21.62 | 6 000 | ~6.9 h | 198 s |
| 3 | 85.23 | 25.40 | 6 000 | ~6.9 h | 201 s |

三个 seed 的 eval `length_mean` = 912 / 951 / 902（满值 1000;play_env `episode_length_s =
20 s` × 50 Hz）—— 策略在混合崎岖地形上能走完约 90–95% 的完整 episode。`success_rate` 为 `null`。
Eval terrain seed = 0（确定性，但 Genesis GPU 浮点非确定性使 `return_mean` 仍有 ~±2 的 run-to-run
方差）。课程在 terrain ~4.5 级自平衡；训练在 6k 收敛、全程无 de-learning（action std 始终维持在
0.3 下限）。**删除 `base_lin_vel` 对性能无影响** —— 与带 `base_lin_vel` 的 v1.0 baseline（下方
admonition）持平。硬件：4× NVIDIA GeForce RTX 4090（一卡一 seed），与 flat sweep 并发跑；`*` 号表示
seed_1 的 eval 与一个并发 flat 训练共享 GPU 故偏长，seed_2/3 在空闲 GPU 上约 200 s。

!!! warning "Eval 需要 `auto_reset` 修复（commit `d56158c`）"
    `genelab eval` 用 play_env（`auto_reset=False`，teleop 用）构建，而 evaluator 依赖 env 自动复位
    收集 episode。未打此 fix 时，崎岖任务 eval 会坍缩成垃圾（`return ≈ -2.6`、`length ≈ 17`，退化的
    ~1 步 episode）。上表数字均为修复后所得。

!!! note "Previous：带 `base_lin_vel`（v1.0 baseline）"

    actor 含 `base_lin_vel`；硬件 H200（`QD_GRAPH=0`，每步约慢 2×）。

    | Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
    |---|---|---|---|---|---|
    | 1 | 83.95 | 23.86 | 6 000 | ~4.8 h | 145 s |
    | 2 | 87.17 | 12.47 | 6 000 | ~4.9 h | 145 s |
    | 3 | 84.41 | 17.67 | 6 000 | ~4.9 h | 145 s |

    `length_mean` = 899 / 966 / 935。

> Maintainer sweep protocol: `genelab train
> Genelab-Velocity-Rough-Unitree-G1-v0 --seeds 1,2,3 --parallel 1 --log_dir
> logs/reference/Genelab-Velocity-Rough-Unitree-G1-v0/<DATE>`，每块 RTX 4090
> 跑一个 seed（4090 集群）。Eval 用 `genelab eval ... --num-envs 64 --episodes
> 100 --seed 0 --out <seed_dir>/eval.json`，并设 `QT_QPA_PLATFORM=offscreen`
> （headless Qt —— 即便 eval 已强制 vis=False，recording 附加项缺这个仍会崩）。

### `Genelab-Tracking-Flat-Unitree-G1-v0`

| Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 138.444 | 0.006 | 30 000 | ~28 h | 227.3 s |
| 2 | 137.980 | 0.008 | 30 000 | ~29 h | 301.6 s |
| 3 | 138.060 | 0.004 | 30 000 | ~29 h | 236.9 s |

Eval `length_mean = 1500.0`。Tracking play_env 默认 `episode_length_s =
1e9` 是为 viewer 无限 playback 设的；`genelab eval` 把它 clamp 到 30 s，
所以 30 s × 50 Hz = 1500 步/ep，全部撞 cap 不 terminate。
三 seed 之间标准差非常紧 —— 收敛策略在 30 s 窗口内稳定跟随 motion clip。
`success_rate` 为 `null`。

!!! note "Previous: v0.4.7"

    | Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
    |---|---|---|---|---|---|
    | 1 | 137.800 | 0.005 | 30 000 | ~20.8 h | 212.8 s |
    | 2 | 138.047 | 0.004 | 30 000 | ~20.6 h | 216.8 s |
    | 3 | 138.122 | 0.007 | 30 000 | ~20.9 h | 216.0 s |

### `Genelab-Velocity-Flat-Unitree-Go1-v0`

| Seed | 最终 `return_mean` | `return_std` | 收敛 iter | 训练 wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 42 | 56.533 | 2.264 | 3 000 | ~55 min | 99.4 s |

单 seed（42，任务 cfg 默认）跑在一张 RTX 5060 Ti 上 —— 见硬件说明；这是
smoke 级参考、不是三 seed sweep，所以没有跨 seed 方差。Eval `length_mean =
1000.0`（play_env `episode_length_s = 20 s` × 50 Hz —— 完整 episode、不摔）。
`success_rate` 为 `null`。Actor 是**纯本体感知**（无基座线速度 —— 真实 Go1 没有
该传感器）；训练时由非对称 critic 拿到特权的真实基座速度。方向分桶 rollout
（256 envs、固定指令）确认跟踪对称 —— 前进/后退/侧移/转向都落在指令速度的
88–97%。

### `Genelab-Velocity-Rough-Unitree-Go1-v0`（进行中）

暂无参考数字。从零训 3k 迭代策略陷入原地站立局部最优 —— 在课程地形上能保持站立，
但只以约 1–2% 的指令速度平移（站立姿态本身就能拿到部分速度跟踪分，从零 3k 迭代
逃不出这个最优；惩罚**不是**原因 —— foot-slip / undesired-contact 项始终可忽略）。
环境接线是对的（非对称 critic + 187 射线 height scan + `terrain_levels` 课程，
都过了 smoke 测试）；问题在训练预算 / bootstrap。要收敛大概需要更大预算（约 6k，
对齐 G1 rough 任务）外加更易的课程 level-0，让从零策略先 bootstrap 出基本行走、
再让地形变难。单独跟踪，正如 G1 rough 任务落地前那样。

### `Genelab-Velocity-Flat-Unitree-Go2W-v0`

Go2-W 是**滑移转向(skid-steer)轮式**四足——前后向轮子既不能侧滚,也无法慢速擦转(静摩擦死区),
横移 / 慢速旋转必须靠**腿式迈步**。现行配置通过两阶段课程 + sim2sim 加固训练**轮腿混合策略**:

1. **第一阶段(`Genelab-Velocity-Flat-Unitree-Go2W-CrabStage1-v0`)** —— 锁轮成刚性圆足
   (`lock_wheels=True`:轮动作 scale 0、阻尼 20)+ Go1 式步态整形(`feet_air_time`、
   `feet_slip`);腿从零学会对称 crab-walk / 迈步转身(6k 迭代)。
2. **第二阶段(本任务)** —— 从第一阶段热启动、轮子解锁(`--checkpoint <stage1>/model_*.pt`,
   4–6k 迭代)。关键配方(均经探针验证过其对应的失败模式):**镜像对称数据增强**(rsl_rl
   `Symmetry`;没有它 PPO 会塌缩成单侧步态——一侧 64/64 完美、镜像侧 64/64 全摔)、
   **L1 跟踪误差项**(`vy_error_l1` −0.5、`wz_error_l1` −0.5;exp 核在某轴被放弃后梯度归零)、
   **vy+wz 门控的 `feet_air_time`**(横移*或*旋转指令下迈步持续有奖励;纯前进走轮滚)、
   **轮阻尼 5.0**(支撑轮指令为零时真能刹住)。sim2sim 加固叠加其上:5 帧观测堆叠(actor 输入
   285 = 57 × 5)、startup 域随机化(轮地摩擦、躯干质量/质心、±20% PD 增益、编码器偏置)、
   每步动作噪声 + 每环境动作延迟(仅训练期)。

| Seed | 最终 `return_mean` | `return_std` | 预算 | 训练 wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 42 | 62.396 | 2.311 | 6k(一阶段)+ 6k + 4k(二阶段) | 共 ~7 h | 21.6 s |

单 seed(42)跑在一张 RTX 5060 Ti 上(见硬件说明)—— smoke 级,不是三 seed sweep。Eval
`length_mean = 1000.0`(完整 episode、50 回合零摔倒)。`success_rate` 为 `null`。
分方向探针(每方向 64 envs、固定指令、确定性、**无自动重置**、逐 env 均值的中位数):
±vy **96%**(左右完全镜像对称)、±wz 93–94%、±vx 94–100%、慢转 wz=0.2 约 77%
(迈步转身;air_time 门控覆盖 wz 之前是 32% 的静摩擦死区)。576 个探针 env 全程零摔倒。

!!! warning "部署:执行器增益 + 堆叠观测"
    导出的 `policy.ts` / `policy.onnx` 期望 **5 帧 frame-major 堆叠观测**(每控制步推入一个
    57 维帧,reset 时回填——schema 见 `policy.*.metadata.json`),且训练时**轮速度增益
    kv = 5.0**(不是 asset 默认的 0.5)—— MuJoCo / 实机侧必须对齐,否则轮响应不一致。

!!! note "沿革:单帧无 DR(51.9)→ 单阶段加固(36.1)→ 轮腿混合(62.4)"
    最初的单帧无 DR 配置得 51.9,但 MuJoCo 迁移差,且横移只会歪身体、原地旋转只会扭身。
    第一轮加固(5 帧堆叠 + DR,单阶段 6k)干净环境得 36.1 —— 鲁棒但横移"跟踪"仍是身体倾斜,
    逐 env 探针暴露了这一点。两阶段课程 + 对称增强 + L1 项同时拿回了真全向跟踪与最高干净回报。

### `GeneLab-Franka-Pick-And-Place-v0` (SAC+HER，demo 预填)

| Seed | 最终 `return_mean` | `return_std` | `success_rate` | 收敛 timestep | 训练 wall-clock | Eval wall-clock |
|---|---|---|---|---|---|---|
| 1 | −6.122 | 13.307 | 0.990 | 2 000 000（预算上限）| ~67 min | 10.7 s |
| 2 | −7.260 | 14.695 | 0.990 | 2 000 000（预算上限）| ~62 min | 10.4 s |
| 3 | −8.790 | 18.275 | 0.970 | 2 000 000（预算上限）| ~64 min | 11.1 s |

Eval `length_mean = 100.0`（固定 episode length）。`success_rate` 由
manipulation 任务的 goal-reach termination 给出。三 seed 平均
`success_rate ≈ 0.983 ± 0.012` —— 比 v0.4.7 那次明显更紧凑：v0.4.7 有一
个 seed 因为末端朝向 drift 只到 0.89；v1.0 三个 seed 全部落在同一个
紧凑区间内。

!!! note "Previous: v0.4.7"

    | Seed | 最终 `return_mean` | `return_std` | `success_rate` | 收敛 timestep | 训练 wall-clock | Eval wall-clock |
    |---|---|---|---|---|---|---|
    | 1 | −19.264 | 33.334 | 0.89 | 2 000 000（预算上限）| ~68 min | 15.3 s |
    | 2 |  −4.626 |  8.297 | 1.00 | 2 000 000（预算上限）| ~63 min | 14.3 s |
    | 3 |  −4.102 |  7.644 | 1.00 | 2 000 000（预算上限）| ~64 min | 18.7 s |

    三 seed 平均 `success_rate ≈ 0.963 ± 0.052`；两个完美 seed 表示策略完全
    收敛，0.89 那个 seed 在比较难的 goal pose 上还会差 ~11 % episode（末端
    朝向 drift）。

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

- **不是** benchmark suite 或排行榜。这里的数字是 GeneLab 自己的复现 sanity
  check。
- **不是** 调参指南。看 `best-practices/rl-experiments` 了解 curriculum / DR
  / reward weight 的选择 —— 那些是这些数字的上游。
