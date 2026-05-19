# Franka 抓取放置

Franka 抓取放置示例展示一个 goal-conditioned manipulation 任务，机器人使用 asset zoo 中的
Franka Panda。一个 4 cm 立方体会在机械臂前方生成，每个环境都会采样一个目标位置，目标可能在地面上，
也可能在空中。奖励沿用 panda-gym 的 dense 形式：末端到方块距离、方块到目标距离，以及进入目标阈值后的成功奖励。

## 任务

| Task id | Action dim | 展示内容 |
|---|---:|---|
| `GeneLab-Franka-Pick-And-Place-v0` | 9 | 直接用 joint-position 控制 7 个手臂关节和 2 个手指关节。 |
| `GeneLab-Franka-Pick-And-Place-Cartesian-v0` | 4 | 通过 differential IK 与 binary gripper 提供 panda-gym 风格的 `(dx, dy, dz, gripper)` 控制。 |

如果要和直接关节控制做 ablation，使用 joint-position 任务。如果要对齐 panda-gym `PandaPickAndPlace`
的 4-DoF action surface，使用 Cartesian 任务。

## 安装并列出

```bash
uv pip install -e examples/franka_pick_and_place
uv run genelab list tasks | grep Franka
```

不安装时：

```bash
PYTHONPATH=examples/franka_pick_and_place/src \
  uv run genelab --import genelab_franka_pick_and_place.tasks list tasks
```

第一次运行可能会下载 Franka MJCF 资产，并构建 Genesis kernel cache。

## 运行 smoke training

Joint-position 变体：

```bash
uv run genelab train GeneLab-Franka-Pick-And-Place-v0 \
  --num-envs 16 \
  --max-iterations 2
```

Cartesian 变体：

```bash
uv run genelab train GeneLab-Franka-Pick-And-Place-Cartesian-v0 \
  --num-envs 16 \
  --max-iterations 2
```

如果想打开 viewer，可在小规模运行时加上 `--vis`。

## 回放 checkpoint

```bash
uv run genelab play GeneLab-Franka-Pick-And-Place-Cartesian-v0 \
  --checkpoint logs/rsl_rl/franka_pick_and_place/<run>/model_2.pt \
  --steps 50
```

请使用与 checkpoint 对应的 task id。两个 action 变体共享 PPO runner 配置，但 policy 网络的输入/输出形状取决于任务的 action space。

## Action 变体

| 变体 | Action terms | Policy 输出 |
|---|---|---|
| Joint position | `JointPositionActionCfg(joint_names=(arm, fingers))` | 9 维 joint-position target，并以机器人默认姿态作为 offset。 |
| Cartesian | `DifferentialIKActionCfg(body_name="hand", joint_names=(arm,))` + `BinaryGripperActionCfg(joint_names=(fingers,))` | 3 维末端位置 delta 加 1 维 gripper scalar。 |

Cartesian 变体会在 Franka articulation 上启用 `requires_jac_and_ik=True`，让 Genesis 提供末端
Jacobian。`DifferentialIKAction` 在每个 control tick 求解一次 damped-least-squares IK，并写入手臂关节目标。
`BinaryGripperAction` 把单个 scalar 映射为两个手指关节的 `closed_pos=0.0` 或 `open_pos=0.04`。

## 代码入口

| 文件 | 作用 |
|---|---|
| `tasks.py` | 注册 robot、env 和两个 task id。 |
| `env_cfg.py` | 构建共享场景，并选择 joint-position 或 Cartesian action 配置。 |
| `mdp.py` | 定义任务自己的 observation、reward、reset 采样和 termination。 |
| `robot.py` | 包装 asset-zoo Franka，并为 Cartesian 控制切换 Jacobian/IK 需求。 |

## 说明

- 方块直接放在 ground plane 上；场景里没有单独的桌子 mesh。
- 机器人 base 初始位置是 `(0, 0, 0)`，不同于 panda-gym 的桌面设置。
- Goal `z` 有 `0.7` 的概率在 `[0, 0.2]` 中均匀采样；否则目标位于地面上的方块高度。
- 成功距离阈值是 `0.05 m`。

## 另见

- [MDP term 参考](../concepts/mdp.md)
- [设计任务](../best-practices/task-design.md)
- [资产库](../concepts/asset_zoo.md)
