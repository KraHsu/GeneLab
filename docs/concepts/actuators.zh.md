# 执行器

执行器是策略 action 与 Genesis joint control 之间的一层。它让机器人配置决定每组关节如何驱动，
而不需要修改 action term 或任务逻辑。

## 为什么需要专门的执行器层

不同机器人需要不同控制假设。简单小车可以用隐式 PD 目标；腿式机器人可能需要力矩限制和 DC motor 饱和。GeneLab 把这些机制放在挂到 `ArticulationCfg` 的 actuator 配置里。

## 内置模型

| 模型 | 行为 |
|---|---|
| `ImplicitPDActuator` | 使用 Genesis/仿真器隐式 PD 控制。 |
| `IdealPDActuator` | 在 Python 中计算 PD torque，并写入 force target。 |
| `DCMotorActuator` | 在 ideal PD 基础上加入电机限制和饱和行为。 |

执行器按配置的关节名或表达式匹配关节组，并向 action term 暴露维度和控制逻辑。

## 设计建议

执行器分组应贴合机器人机构。若手臂、手、底座关节需要不同增益、限制或 action scale，不要放进一个巨大的 actuator。

## 继续阅读

- [设计任务](../best-practices/task-design.md)
- [API 参考](../api/reference.md)
