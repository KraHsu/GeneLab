# MDP term 参考

`genelab.mdp` 是可复用 term 库。它本身不定义任务；任务配置从这个库里选择函数和类，并接到 manager 中。

## Actions 与 commands

| 区域 | 公开组件 |
|---|---|
| Actions | `JointPositionActionCfg`、`JointPositionAction` |
| 速度命令 | `UniformVelocityCommandCfg`、`UniformVelocityCommand` |
| Motion 命令 | `MotionCommandCfg`、`MotionCommand`、`MotionLoader` |

action 把策略输出转换为仿真控制。command 持有 observation 和 reward 可读取的采样目标。

## Observations

常见 observation 函数包括 base velocity、projected gravity、relative joint position/velocity、
last action、generated commands、sensor data、contact 特征、terrain height scan 和 motion-tracking 状态。

observation 函数应返回 `(num_envs, d)` 或 `(num_envs,)` tensor。Observation manager 负责可选噪声、缩放、裁剪和 group concat。

## Rewards 与 terminations

reward 函数覆盖速度跟踪、action 平滑、关节加速度、姿态、关节限制、足端 clearance、slip、air time、
self collision、角动量和 motion-tracking 误差。termination 函数覆盖 time-out、姿态、root height 和 motion-tracking 失败。

reward 函数应返回 `(num_envs,)`；termination 函数应返回 `(num_envs,)` bool tensor。

## Events、curricula、metrics、noise

| 区域 | 示例 |
|---|---|
| Events | `reset_root_state_uniform`、`reset_joints_to_default`、`push_by_setting_velocity` |
| Curricula | `terrain_levels_vel`、`commands_vel` |
| Metrics | `mean_action_acc`、`angular_momentum_mean`、`air_time_mean`、`slip_velocity_mean` |
| Noise | `Unoise`、`Gnoise` |
| Domain randomization | `mdp.dr.body`、`mdp.dr.joint`、`mdp.dr.geom` |

## 继续阅读

- [Manager 与 MDP term](managers.md)
- [设计任务](../best-practices/task-design.md)
- [API 参考](../api/reference.md)
