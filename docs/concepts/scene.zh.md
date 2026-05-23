# 场景与实体

`InteractiveScene` 是 GeneLab 的 Genesis scene owner。它把声明式配置转换成 live Genesis handle，
并向环境其他部分暴露 Isaac Lab 风格的 entity wrapper。

## Scene 边界

`InteractiveSceneCfg` 描述应该存在什么：env spacing、entities、terrain、sensors、recordings、
viewer interaction、batch rendering。`InteractiveScene` 持有实际存在的东西：Genesis `Scene`、
articulation、rigid object、sensor、terrain importer、recorder bridge 和 viewer 状态。

这种分离让配置可序列化，也让任务能在 Genesis 启动前被检查。

## Entities

| Entity | 用途 |
|---|---|
| `Articulation` | 机器人 wrapper，包含 joint/link 名、默认 joint state、limits、刷新后的 `RobotState`。 |
| `RigidObject` | 非关节物体 wrapper。 |
| `RobotState` | observation、reward、sensor、event 读取的 batched tensor。 |

`ManagerBasedRlEnv` 把配置中的机器人作为 `"robot"` articulation 加入。实时机器人数据通过具名
entity 表访问，例如用 `env.articulations["robot"].data` 读取 `RobotState`，用
`env.articulations["robot"].joint_names` 读取关节元数据。环境也暴露 `env.scene` 供 scene 级访问。

## 为什么需要 wrapper

Genesis API 和 Isaac Lab 风格任务代码使用的词汇不同。wrapper 隔离这种差异：MDP term 读取稳定的 GeneLab 属性，后端集成负责 Genesis 细节。

## 继续阅读

- [模块地图](../reference/module-map.md)
- [传感器](sensors.md)
- [执行器](actuators.md)
