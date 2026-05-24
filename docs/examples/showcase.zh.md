# Showcase

`examples/genelab_showcase` 包含面向单个框架能力的聚焦任务。需要查看某个子系统的可运行示例，而不是完整机器人任务时，用它。

## 能力

| 区域 | 展示内容 |
|---|---|
| 传感器 | Body velocity、IMU-like state、contact、ray-cast、terrain height、joint force/torque。 |
| 接触 | 接触力、air time、landing 和 slip metric。 |
| 地形 | 生成式地形网格和 height scan。 |
| Curriculum | terrain-level progression。 |
| 执行器 | 切换 actuator 模型（IdealPD、MLP-residual）和 action 行为。 |
| Recording | 实时曲线、文件、面向视频的 recording 配置。 |

## 运行

```bash
uv pip install -e examples/genelab_showcase
uv run genelab list tasks
uv run genelab play <showcase-task-id> --vis --steps 300
```

用 `genelab info <showcase-task-id>` 查看每个 showcase task 的准确 override 路径。

## 另见

- [传感器](../concepts/sensors.md)
- [数据录制与实时绘图](../concepts/recording.md)
- [地形](../concepts/terrains.md)
