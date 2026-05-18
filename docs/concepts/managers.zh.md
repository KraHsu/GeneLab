# Manager 与 MDP term

GeneLab 使用 Isaac Lab 风格的 manager-based 环境模式：环境负责仿真循环，MDP 行为拆成命名 term，
由专门 manager 管理。

## 为什么拆分 MDP

机器人学习环境通常包含很多关注点：

- action 解码
- command 采样
- observation
- reward
- termination
- reset 和 interval event
- curriculum
- metric

如果全部塞进 `step()`，任务会很难检查和 override。manager-based 配置让每个关注点都有名字、类型和可发现路径。

## manager 集合

| Manager | 配置字段 | 作用 |
|---|---|---|
| `ActionManager` | `actions_cfg` | 把策略 action 转成仿真目标或力矩。 |
| `CommandManager` | `commands_cfg` | 维护采样目标，例如目标速度或参考 motion。 |
| `ObservationManager` | `observations_cfg` | 计算 `policy`、`critic` 等 observation group。 |
| `RewardManager` | `rewards_cfg` | 计算带权 reward term 和 episode 摘要。 |
| `TerminationManager` | `terminations_cfg` | 区分 terminated 和 time-out 条件。 |
| `EventManager` | `events_cfg` | 执行 startup、reset、interval 随机化或扰动。 |
| `CurriculumManager` | `curriculum_cfg` | 在 reset 边界调整难度或任务状态。 |
| `MetricsManager` | `metrics_cfg` | 记录非 reward 诊断指标。 |

## 运行时顺序

`ManagerBasedRlEnv` 先构建 Genesis scene，再创建 manager、绑定传感器、执行 startup event、
跑初始 reset。每个 env step 中，它处理 action，推进 `decimation` 个 physics tick，刷新状态，
更新传感器，计算 command、event、reward、metric、termination，reset 已完成 env，最后计算 observation。

这个顺序很重要，因为 term 读取共享状态。例如 reward term 看到的是 action 和 sensor update 之后的状态；
reset event 会在下一次 observation 发出前运行。

## term 名称是接口的一部分

term 名会出现在 override 路径和日志中。名为 `track_lin_vel` 的 reward 可以从 CLI 修改，也会出现在 episode 摘要里。稳定命名让实验更容易比较。

## 继续阅读

- [设计任务](../best-practices/task-design.md)
- [MDP term 参考](mdp.md)
- [API 参考](../api/reference.md)
