# 传感器

Genesis 不会通过 MJCF sensor block 暴露所有信号，因此 GeneLab 提供后端感知的传感器抽象。
传感器产出带缓存的数据，供 observation、reward、recording 和自定义代码消费。

## 生命周期

传感器声明在 `InteractiveSceneCfg.sensors` 中。`ManagerBasedRlEnv` 在 scene 存在后构建并绑定它们。
每个控制步会使缓存失效；第一次访问 `sensor.data` 时计算新输出。reset 会清理每 env 状态。

这种惰性缓存保证同一个控制步内多次 observation 和 reward 读取一致。

## 内置传感器族

| 传感器 | 用途 |
|---|---|
| `BodyVelocitySensor` | offset 处 link 线速度或角速度。 |
| `IMUSensor` | 姿态、projected gravity、线/角加速度。 |
| `CameraSensor` | 挂到 link 上的 RGB-D 相机。 |
| `ContactSensor` | 每 link 接触力和可选 air/contact time。 |
| `FrameTransformerSensor` | target frame 相对 source frame 的姿态。 |
| `RayCastSensor` | grid、ring、hemisphere ray pattern。 |
| `TerrainHeightSensor` | 地形感知 locomotion 的向下 height scan。 |
| `SelfContactSensor` | 自碰撞/自接触信号。 |
| `RootAngularMomentumSensor` | root angular momentum 诊断。 |

## MDP term 中的传感器数据

observation term 可以通过 `mdp.sensor_data` 或自定义 callable 读取传感器。reward 和 metric 可以直接访问 `env.sensors[name].data`。

保持传感器名称稳定；它们会成为配置路径、recording source 和 term params。

## Camera 注意事项

并行 RGB-D 渲染需要 CUDA 和 `InteractiveSceneCfg.batch_render=True`。把 camera task 当成不同于轻量状态 RL task 的硬件 profile。

## 继续阅读

- [数据录制与实时绘图](recording.md)
- [模块地图](../reference/module-map.md)
- [API 参考](../api/reference.md)
