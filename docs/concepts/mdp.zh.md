# MDP term 参考

`genelab.mdp` 是可复用 term 库。它本身不定义任务；任务配置从这个库里选择函数和类，并接到 manager 中。

## 动作与命令

| 区域 | 公开组件 |
|---|---|
| Actions | `JointPositionActionCfg`、`JointPositionAction` |
| Cartesian actions | `DifferentialIKActionCfg`、`DifferentialIKAction`、`BinaryGripperActionCfg`、`BinaryGripperAction` |
| 速度命令 | `UniformVelocityCommandCfg`、`UniformVelocityCommand` |
| Motion 命令 | `MotionCommandCfg`、`MotionCommand`、`MotionLoader` |

action 把策略输出转换为仿真控制。command 持有 observation 和 reward 可读取的采样目标。

### 动作 term

| Term | Action dim | 说明 |
|---|---:|---|
| `JointPositionAction` | 匹配到的关节数 | 把 policy 输出映射为选中关节的 joint-position target。 |
| `DifferentialIKAction` | 3 或 6 | 用 damped-least-squares Jacobian IK 把末端 delta 转成手臂关节目标。 |
| `BinaryGripperAction` | 1 | 从一个 scalar action 把匹配到的手指关节切到 open 或 closed 目标。 |

`DifferentialIKActionCfg.body_name` 选择要控制的 link，`joint_names` 选择参与 IK 求解的手臂关节。
当 `use_orientation=False` 时，policy 输出 `(dx, dy, dz)`；当 `use_orientation=True` 时，policy 输出
`(dx, dy, dz, droll, dpitch, dyaw)`，其中姿态部分是 axis-angle delta。`scale` 限制每个 control step
的物理 delta，`damping` 用来正则化 IK 求解，`max_delta_joint` 用来限制每个关节的单步更新。

使用 `DifferentialIKAction` 前，机器人 articulation 必须设置 `requires_jac_and_ik=True`，否则 Genesis
不会分配求解器需要的 Jacobian 数据。

`BinaryGripperActionCfg` 使用 `threshold` 在 `closed_pos` 与 `open_pos` 之间切换。它通常和 Cartesian
手臂 term 组合使用；例如 Franka Cartesian 抓取放置任务把 `DifferentialIKAction(body_name="hand")`
和 `BinaryGripperAction` 组合成 4 维 `(dx, dy, dz, gripper)` action space。

## 观测

常见 observation 函数包括 base velocity、projected gravity、relative joint position/velocity、
last action、generated commands、sensor data、contact 特征、terrain height scan 和 motion-tracking 状态。

observation 函数应返回 `(num_envs, d)` 或 `(num_envs,)` tensor。Observation manager 负责可选噪声、缩放、裁剪和 group concat。

## 奖励与终止

reward 函数覆盖速度跟踪、action 平滑、关节加速度、姿态、关节限制、足端 clearance、slip、air time、
self collision、角动量和 motion-tracking 误差。termination 函数覆盖 time-out、姿态、root height 和 motion-tracking 失败。

reward 函数应返回 `(num_envs,)`；termination 函数应返回 `(num_envs,)` bool tensor。

## 事件、课程、指标、噪声

| 区域 | 示例 |
|---|---|
| Events | `reset_root_state_uniform`、`reset_joints_to_default`、`push_by_setting_velocity` |
| Curricula | `terrain_levels_vel`、`commands_vel` |
| Metrics | `mean_action_acc`、`angular_momentum_mean`、`air_time_mean`、`slip_velocity_mean` |
| Noise | `Unoise`、`Gnoise` |
| Domain randomization | `mdp.dr.body`、`mdp.dr.joint`、`mdp.dr.geom` |

`NoiseCfg` 的 canonical home 是 `genelab.contracts`，这样 observation manager 可以标注 noise 而无需
导入 `genelab.mdp`。具体噪声模型仍可从 `genelab.mdp.noise` 和公开 facade 导入。

## 继续阅读

- [Manager 与 MDP term](managers.md)
- [Franka 抓取放置](../examples/franka-pick-and-place.md)
- [设计任务](../best-practices/task-design.md)
- [API 参考](../api/reference.md)
