# TemperatureGridSensor

粗粒度温度网格传感器，底层是 `genesis.sensors.TemperatureGrid`。

在机器人某个 link 上挂一个规则的 `(nx, ny, nz)` 温度采样网格。适用于
接触摩擦发热相关任务的低成本温度代理 —— 摩擦升温的末端执行器、尾气
plume、温度感知抓取。本封装是最简版本，仅暴露分辨率和环境温度；
Genesis 更细的 surface（按材料的 `properties_dict`、`heat_generation`、
`convection_coefficient`）暂未暴露。

## 配置

`TemperatureGridSensorCfg` 继承 `SensorCfg`。全部字段使用关键字参数。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | `str` | `""` | 继承自 `SensorCfg`，场景内唯一。 |
| `entity_name` | `str` | `"robot"` | 继承自 `SensorCfg`，挂载网格的实体。 |
| `link_name` | `str` | `""` | 挂载网格的 link。必填。 |
| `grid_size` | `tuple[int, int, int]` | `(1, 1, 1)` | 在 link 本地坐标系下的 `(nx, ny, nz)` 分辨率。 |
| `ambient_temperature` | `float \| None` | `None` | 初始温度（摄氏度）。`None` 时使用 Genesis 按材料的默认值。 |
| `history_length` | `int` | `0` | 透传给 Genesis。`0` 表示只读当前帧；`> 0` 分配 ring buffer，`data.temperature` 多一个 history 维度。 |

## 数据

`TemperatureGridData.temperature` 形状为
`(num_envs, [history,] nx, ny, nz)`，单位摄氏度。

## 示例

```python
from genelab.configs import InteractiveSceneCfg
from genelab.sensor import TemperatureGridSensorCfg

scene_cfg = InteractiveSceneCfg(
    sensors=(
        TemperatureGridSensorCfg(
            name="gripper_thermals",
            link_name="right_gripper",
            grid_size=(4, 4, 2),
            ambient_temperature=22.0,
        ),
    ),
)
```

## 另见

- [KinematicContactSensor](kinematic_contact.md)
- [Concepts — Sensors](../../concepts/sensors.md)
