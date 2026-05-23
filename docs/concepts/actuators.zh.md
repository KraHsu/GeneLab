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
| `MlpResidualActuator` | 在 `DCMotorActuator` 基础上叠加 TorchScript 残差力矩模型。 |

执行器按配置的关节名或表达式匹配关节组，并向 action term 暴露维度和控制逻辑。

## 使用 `MlpResidualActuatorCfg`

当机器人已有可用的 DC motor 模型，但真机日志显示稳定的力矩跟踪偏差时，使用
`MlpResidualActuatorCfg`。执行器从 `network_file` 加载 TorchScript 模块，并把网络输出加到
DC motor 力矩上：

```python
from genelab.actuator import MlpResidualActuatorCfg

robot_cfg.actuators["legs"] = MlpResidualActuatorCfg(
    target_names_expr=(".*_hip_joint", ".*_knee_joint", ".*_ankle_joint"),
    stiffness=40.0,
    damping=1.0,
    effort_limit=120.0,
    velocity_limit=30.0,
    saturation_effort=120.0,
    action_scale=0.25,
    network_file="assets/actuators/leg_residual.pt",
    residual_scale=0.5,
)
```

TorchScript 模块接收最后一维为 `[target_pos - joint_pos, joint_vel]` 的 tensor，并为每个关节
返回一个残差力矩。第一层为 `nn.Linear(2, hidden)` 的标准 MLP 满足这个接口。把
`network_file` 设为 `None` 时，配置形状保持不变，但行为退化为普通 `DCMotorActuator`。

`velocity_limit` 是必填项，因为 `MlpResidualActuatorCfg` 继承 DC motor 的 torque-speed 模型。
`effort_limit` 或 `saturation_effort` 必须定义最终力矩预算；残差叠加后会被重新 clamp 回这个预算。

## 设计建议

执行器分组应贴合机器人机构。若手臂、手、底座关节需要不同增益、限制或 action scale，不要放进一个巨大的 actuator。

## 继续阅读

- [设计任务](../best-practices/task-design.md)
- [API 参考](../api/reference.md)
