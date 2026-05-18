# Unitree G1

`examples/unitree/` 在平地上提供 2 个 PPO 任务，机器人是 Unitree G1 人形。两个任务都
对标 `mjlab` 的同名 recipe，移植到 Genesis 上。

## 任务列表

| Task id | 问题 |
|---------|---------|
| `Genelab-Velocity-Flat-Unitree-G1-v0` | 速度跟踪基线——跟踪 body 系的命令 twist。 |
| `Genelab-Tracking-Flat-Unitree-G1-v0` | 动作模仿——按 body 跟踪录制 clip (BeyondMimic 风格)。 |

## 安装

`torch-*` extra 按本机硬件挑。

```bash
uv sync --extra torch-cu128
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

tracking 任务消费 NPZ motion clip (key：`joint_pos`、`joint_vel`、`body_pos_w`、
`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w` —— mjlab `csv_to_npz` 产出的 schema)。
默认 clip 是 LAFAN1 retargeted 的 `dance1_subject2` NPZ，首次使用时由
`genelab.asset_zoo.unitree_g1_motions.g1_lafan1_dance1_subject2` 拉取并缓存到
`.cache/assets/g1_lafan1_dance1_subject2/<md5>/`，无需手动下载，example 开箱即用。

```bash
# 逐帧回放参考 clip (不接策略；机器人每步贴到 motion 当前帧)。
# 训练前验证 env 接线是否正确的标准方式。
uv run python -m genelab_unitree.replay_motion

# 训练。
uv run genelab train Genelab-Tracking-Flat-Unitree-G1-v0 \
    --num-envs 4096 --max-iterations 30000

# 回放训练好的策略。
uv run genelab play Genelab-Tracking-Flat-Unitree-G1-v0 \
    --agent trained \
    --checkpoint logs/rsl_rl/g1_tracking_flat/<run>/model_30000.pt
```

`--agent` 取 `zero`、`random` 或 `trained`。不写 `--agent` 时，`--checkpoint` 存在则默认
`trained`，否则 `zero`。`zero` 和 `random` 都只在 reset 时把机器人摆到 clip 第 0 帧、
之后施加零 / 随机力矩，机器人只会倒下，不会跟随动作。要做无策略的参考回放，用上面的
`replay_motion` 脚本。

### 替换 clip

默认 `motion_file` 在 `tracking_env_cfg.unitree_g1_tracking_env_cfg` 里硬编码。把
`g1_lafan1_dance1_subject2()` 替换成符合上面 schema 的任意 NPZ 路径即可；若新 clip 的
body / joint 轴顺序不同，记得同步更新 `MotionCommandCfg` 的 `motion_body_order` /
`motion_joint_order`。[`genelab-assets`](https://github.com/KraHsu/genelab-assets) 仓库
里附带 `unitree_g1/motions/scripts/convert.sh`，把任意 G1 retargeted CSV 喂给 mjlab 的
`csv_to_npz` 做一次正运动学回放。

捆绑 clip 继承上游许可 (CC BY-NC-ND 4.0 —— 仅供非商业研究，需署名)；详情见 assets 仓库
里的 `unitree_g1/motions/LICENSE.NOTICE`。

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

- G1 的 MJCF 与 STL mesh 在首次使用时由 `genelab.asset_zoo.unitree_g1.UnitreeG1Cfg`
  从 `genelab-assets` 仓库拉取并缓存到 `.cache/`。源自 Unitree 的 `mujoco_menagerie`
  release，license 详见上游仓库。
- 默认 motion NPZ (`dance1_subject2`) 同样从 `genelab-assets` 首次使用时拉取。clip
  内的 body / joint 轴按 mjlab MJCF 的 DFS 顺序存储；env config 通过 `MotionCommandCfg`
  的 `motion_body_order` / `motion_joint_order` 把两轴重排到 Genesis 的遍历顺序。
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

- [倒立摆](inverted-pendulum.md)
- [五指手](wuji-hand.md) —— 更小规模的固定轨迹回放示例
- [Actuators](../concepts/actuators.md)
- [Asset zoo](../concepts/asset_zoo.md)
- [传感器](../concepts/sensors.md)
- [play 与 train CLI](../cli/play-train.md)
