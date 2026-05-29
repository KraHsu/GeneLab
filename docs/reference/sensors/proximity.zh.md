# ProximitySensor

表面距离探针，底层是 `genesis.sensors.SurfaceDistanceProbe`。

一组点状探针刚性挂在父 link 上，每帧报告到一组**被跟踪**目标 link 的最近表面
距离。典型用法：proximity gating、软自碰撞规避、地形/另一 link 作为目标的
time-to-contact 估计。

## 配置

`ProximitySensorCfg` 继承 `SensorCfg`。全部字段使用关键字参数。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | `str` | `""` | 继承自 `SensorCfg`，场景内唯一。 |
| `entity_name` | `str` | `"robot"` | 继承自 `SensorCfg`，挂载探针的实体。 |
| `link_name` | `str` | `""` | 探针所在的父 link。必填，必须在 `entity.link_names` 中。 |
| `probe_local_pos` | `tuple[tuple[float, float, float], ...]` | `((0.0, 0.0, 0.0),)` | 探针在父 link 本地坐标系下的位置。 |
| `probe_radius` | `float` | `10.0` | 每个探针的搜索半径；同时也是无目标在范围内时返回的饱和值。 |
| `track_link_names` | `tuple[str, ...]` | `()` | 非空的待测距 link 集合，对应同实体的 `link_names`。 |
| `history_length` | `int` | `0` | 透传给 Genesis。`0` 表示只读当前帧；`> 0` 分配 ring buffer，`data.distance` 多一个 history 维度。 |

!!! warning "`track_link_names` 必填"
    Genesis 拒绝构造 `track_link_idx` 为空的 `SurfaceDistanceProbe`。封装层在
    最早期就抛 `ValueError`，提示信息一致；务必填入要测距的目标 link。

## 数据

`ProximityData.distance` 形状为 `(num_envs, [history,] num_probes)`，单位米，
范围外的探针读数被 clip 到 `cfg.probe_radius`。

## 示例

```python
from genelab.configs import InteractiveSceneCfg
from genelab.sensor import ProximitySensorCfg

scene_cfg = InteractiveSceneCfg(
    sensors=(
        ProximitySensorCfg(
            name="foot_proximity",
            link_name="left_foot",
            probe_local_pos=((0.0, 0.0, -0.02),),
            probe_radius=0.5,
            track_link_names=("terrain",),
        ),
    ),
)
```

## See also

- [KinematicContactSensor](kinematic_contact.md)
- [Concepts — Sensors](../../concepts/sensors.md)
