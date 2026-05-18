# 五指手

`examples/genelab_examples/src/genelab_examples/wuji_hand/` 是一个 play-only 的 Genesis 演示，
让 Wuji 五指灵巧手回放一段固定的 `.npy` 轨迹。和[魔方示例](rubiks-cube.md)一样，它**不是**
RL 任务——直接构造 Genesis `Scene`，每个关节用 PD 控制器跟踪下一帧轨迹目标。它的意义在于
展示用最少的 GeneLab + Genesis 胶水��码就能让机器人按预录动作走完。

## 任务列表

| Task id | 问题 |
|---------|------|
| `GeneLab-Wuji-Hand-Playback-v0` | 加载右手 MJCF，将每个 `right_fingerN_jointM` 与 Genesis DOF 对齐，然后通过 `control_dofs_position` 逐帧重放一段 20 列轨迹。 |

## 安装

```bash
uv sync --extra torch-cu128
uv pip install -e examples/genelab_examples

uv run genelab list tasks | grep Wuji
# -> GeneLab-Wuji-Hand-Playback-v0
```

或免安装运行：

```bash
PYTHONPATH=examples/genelab_examples/src \
  uv run genelab --import genelab_examples.tasks list tasks
```

## 运行

```bash
# 默认轨迹：包内自带的 20 自由度挥手动作。
uv run genelab play GeneLab-Wuji-Hand-Playback-v0 --vis --steps 600

# 关闭周期性硬复位，让轨迹连续播放。
uv run genelab play GeneLab-Wuji-Hand-Playback-v0 --vis --env.reset_interval 0

# 切到左手 MJCF（默认只带 right）。
uv run genelab play GeneLab-Wuji-Hand-Playback-v0 --vis --env.robot.side left
```

`reset_interval` 是每隔多少 Genesis 步把手重置回零位姿态的间隔，`0` 表示关闭。默认的
`wave.npy` 与包源码同目录。

## 这个示例演示了什么

| GeneLab 能力 | 出现位置 | 概念文档 |
|---|---|---|
| 注册 play-only 任务 | `tasks.py` | [Registry](../concepts/registry.md) |
| 基于磁盘 MJCF 资产目录注册机器人 | `robots.py`、`wuji_hand/assets.py:resolve_mjcf_path` | [Registry](../concepts/registry.md)、[Asset zoo](../concepts/asset_zoo.md) |
| 嵌套 `SimulationCfg`、默认 `Path` 字段的 config dataclass | `wuji_hand/config.py` | [Configs](../concepts/configs.md) |
| 关节名 → Genesis DOF 下标的映射，用于轨迹重放 | `wuji_hand/sim.py:build_joint_mapping`、`JointMapping` dataclass | 无（示例自带的辅助器） |
| 从 `.npy` 加载轨迹 + 按 DOF 上下限逐关节裁剪 | `wuji_hand/assets.py:load_trajectory`、`wuji_hand/sim.py:trajectory_target` | 与 [Unitree G1](unitree-g1.md) 的动作模仿模式相同 |
| 运行时按需设置 PD 增益和力矩范围 | `wuji_hand/sim.py:apply_wuji_gains` | [Actuators](../concepts/actuators.md) —— 与"在配置里直接声明 actuator"模式形成对比 |

## 代码走读

整个包约 400 行，每个文件只做一件事：

- **`wuji_hand/config.py`（24 行）** —— `WujiEnvCfg`（继承 `ManagerBasedEnvCfg`）和
  `WujiRobotCfg`。默认 `dt=0.01`、`steps=0`（viewer 死循环）、`reset_interval=500`、
  `side="right"`。全部可以通过点号 flag 覆写（`--env.simulation.dt 0.005`、
  `--env.reset_interval 0`、`--env.robot.side left`）。
- **`wuji_hand/assets.py`（74 行）** —— 路径解析与轨迹加载。
  - `wuji_joint_names(side)` 按轨迹列顺序返回 20 个关节名
    （`{side}_fingerN_jointM`，`N=1..5, M=1..4`）。
  - `resolve_mjcf_path(desc_dir, side)` 顺序尝试三种候选路径，方便用户接入自己的 MJCF 目录布局。
  - `load_trajectory(path)` 校验形状（2D、≥20 列），返回 `float32`。
- **`wuji_hand/sim.py`（280 行）** —— 重放循环。建议按顺序读：
    1. `WujiHandRunConfig` —— 扁平的运行时配置，`__post_init__` 做基本校验。
    2. `build_joint_mapping(entity, side)` —— 对每个轨迹列，按名字查找匹配的 Genesis
       关节并记录其本地 DOF 下标。找不到的关节只打 warning 跳过，不直接报错。
    3. `apply_wuji_gains` —— 给映射上的每个 DOF 设 `kp=0.8`、`kv=0.04`，并把 MJCF
       声明的力矩范围复制到 Genesis 的运行时 force range（仅当上下限都有限时；
       `forcerange="0 0"` 或未设值的 MJCF 不动）。
    4. `run_wuji_hand(config)` —— 初始化 Genesis，按高自由度手的自碰撞需求调好 rigid
       options，建场景，做完关节映射后进入主循环：用裁剪过的轨迹帧每步调一次
       `entity.control_dofs_position(target, mapping.dof_indices)`。
- **`envs.py:WujiHandPlaybackEnv.play`** —— 薄包装，把 `WujiEnvCfg` 的字段拷到一个
  `WujiHandRunConfig`，再调 `run_wuji_hand`。**没有 `ManagerBasedRlEnv`**——和魔方示例
  一样，这个任务没有 obs/action/reward 图。
- **`tasks.py`** —— 把 `GeneLab-Wuji-Hand-Playback-v0` 注册为 `trainable=False` 的任务。

## Smoke test

```bash
PYTHONPATH=examples/genelab_examples/src \
  uv run genelab --import genelab_examples.tasks play GeneLab-Wuji-Hand-Playback-v0 --steps 5
```

5 步运行足以验证 MJCF 解析、关节映射和一帧轨迹的下发。首次启动会触发 Genesis
kernel 编译。

## See also

- [魔方](rubiks-cube.md) —— 同一个 extension 里的另一个 play-only 示例。
- [Unitree G1](unitree-g1.md) —— 更大规模的动作回放示例（LAFAN1 NPZ、整人形机器人）。
- [Registry](../concepts/registry.md) —— task 和 robot 是怎样接入注册表的。
- [Configs](../concepts/configs.md) —— `--env.robot.side left` 走的 dataclass 覆写机制。
