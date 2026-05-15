# Unitree G1

`examples/unitree/` 在平地上提供 2 个 PPO 任务，机器人是 Unitree G1 人形。两个任务都
对标 `mjlab` 的同名 recipe，移植到 Genesis 上。

## 任务列表

| Task id | 问题 |
|---------|---------|
| `Genelab-Velocity-Flat-Unitree-G1-v0` | 速度跟踪基线——跟踪 body 系的命令 twist。 |
| `Genelab-Tracking-Flat-Unitree-G1-v0` | 动作模仿——按 body 跟踪录制 clip (BeyondMimic 风格)。 |

## 安装

扩展依赖 `rl` extra (rsl_rl)。`torch-*` extra 按本机硬件挑。

```bash
uv sync --extra rl --extra torch-cu128
uv pip install -e examples/unitree

uv run genelab list tasks
# -> Genelab-Velocity-Flat-Unitree-G1-v0
# -> Genelab-Tracking-Flat-Unitree-G1-v0
```

## 速度跟踪

```bash
uv run genelab train Genelab-Velocity-Flat-Unitree-G1-v0 \
    --num-envs 4096 --max-iterations 1500

uv run genelab play  Genelab-Velocity-Flat-Unitree-G1-v0 \
    --checkpoint logs/rsl_rl/g1_velocity_flat/<run>/model_1500.pt --vis
```

## 动作模仿

tracking 任务需要一段 mjlab NPZ schema 的 motion clip (key：`joint_pos`、`joint_vel`、
`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`)。先用
`mjlab.scripts.csv_to_npz` 把 CSV 转出来，再通过 `--env.commands.motion.motion_file`
传进去。

```bash
# 仅看 clip 本身 (机器人复位到 clip 帧、零力矩)。
uv run genelab play Genelab-Tracking-Flat-Unitree-G1-v0 \
    --agent zero \
    --env.commands.motion.motion_file path/to/clip.npz \
    --vis

# 随机动作 sanity check，参考姿态附近的扰动。
uv run genelab play Genelab-Tracking-Flat-Unitree-G1-v0 \
    --agent random \
    --env.commands.motion.motion_file path/to/clip.npz \
    --vis

# 训练。
uv run genelab train Genelab-Tracking-Flat-Unitree-G1-v0 \
    --env.commands.motion.motion_file path/to/clip.npz \
    --num-envs 4096 --max-iterations 30000

# 回放训练好的策略。
uv run genelab play Genelab-Tracking-Flat-Unitree-G1-v0 \
    --agent trained \
    --checkpoint logs/rsl_rl/g1_tracking_flat/<run>/model_30000.pt \
    --env.commands.motion.motion_file path/to/clip.npz
```

`--agent` 取 `zero`、`random` 或 `trained`。不写 `--agent` 时，`--checkpoint` 存在则默认
`trained`，否则 `zero`。

## 长时间训练

多小时的训练 (tracking 任务默认 `max_iterations=30_000`，在 8×H200 上要数小时墙钟)，
启动训练**前**把 PyTorch 内存分配器切到可增长 segment，能显著减少碎片化导致的逐步降速。

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

如果训练时长仍然在 iteration 间漂移，重跑时加 `GENELAB_PROFILE=1` 捕一段 profiler。
trace 落在 `logs/torch_profile/`，可用 `tensorboard --logdir logs/torch_profile` 看。
在训练初期和数小时后各抓一段，对比 per-section 累计时间。可选 env var：
`GENELAB_PROFILE_OUT`、`GENELAB_PROFILE_WAIT`、`GENELAB_PROFILE_WARMUP`、
`GENELAB_PROFILE_ACTIVE`、`GENELAB_PROFILE_REPEAT` (见 `src/genelab/rl/_profiler.py`)。
分布式启动时只 rank 0 写 trace。

!!! tip "Smoke-test 预算"
    `--num-envs 64 --max-iterations 5` 跑 5–10 个 iteration 就能验证 env 接线 OK。这种
    规模下 reward 信号噪声大；速度任务真正收敛需要上面 1500 iteration 的预算。

## 备注

- G1 的 MJCF 与 STL mesh 在 `examples/unitree/assets/g1/` 下随仓库分发。源自 Unitree 的
  `mujoco_menagerie` release，license 详见上游仓库。
- 动作模仿任务去掉了 mjlab 的 adaptive-bin 失败采样；只接了 `start` 与 `uniform` 两种
  采样模式。自碰惩罚也省略，因为 GeneLab 还没有接触对 sensor 抽象——env 仍然惩罚动作
  变化率与关节越界。
- tracking 任务的 `policy` / `critic` observation group 不同：critic 看得到特权的 per-body
  位姿 / 朝向特征，actor 看不到。

## Log 输出

两个任务都写到 `logs/rsl_rl/<experiment>/<timestamp>/`，与倒立摆任务一致：

- `params/env.json` 与 `params/agent.json` — 运行时冻结下来的 cfg。
- `model_<iter>.pt` — 每 `save_interval` 个 iteration 一份 checkpoint。
- TensorBoard event 文件与 checkpoint 同目录。

## See also

- [Inverted Pendulum](inverted-pendulum.md)
- [Actuators](../concepts/actuators.md)
- [Play and Train CLI](../cli/play-train.md)
