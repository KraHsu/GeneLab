# KinematicContactSensor

点接触探针，底层是 `genesis.sensors.ContactProbe`。

一组点状探针刚性挂在父 link 上，每帧报告布尔接触。Genesis 在内部按
`cfg.contact_threshold` 阈值化（默认 `1e-4` m 穿透深度），返回已经阈值化
的布尔 mask。适用于"机器人上这一点有没有碰到东西"的场景 —— 不引入可变形
elastomer tactile 栈。

## 配置

`KinematicContactSensorCfg` 继承 `SensorCfg`。全部字段使用关键字参数。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | `str` | `""` | 继承自 `SensorCfg`，场景内唯一。 |
| `entity_name` | `str` | `"robot"` | 继承自 `SensorCfg`，挂载探针的实体。 |
| `link_name` | `str` | `""` | 探针所在的父 link。必填。 |
| `probe_local_pos` | `tuple[tuple[float, float, float], ...]` | `((0.0, 0.0, 0.0),)` | 探针在父 link 本地坐标系下的位置。 |
| `probe_radius` | `float` | `0.01` | 探针球半径，单位米（Genesis 默认）。 |
| `contact_threshold` | `float` | `1e-4` | 接触深度阈值（米），Genesis 据此判定接触，封装层据此推出 `data.in_contact`。 |
| `history_length` | `int` | `0` | 透传给 Genesis。`0` 表示只读当前帧；`> 0` 分配 ring buffer，`data.depth` / `data.in_contact` 多一个 history 维度。 |

## 数据

| 字段 | 类型 | 形状 |
|---|---|---|
| `in_contact` | `torch.Tensor` (bool) | `(num_envs, [history,] num_probes)` —— Genesis 内部阈值化后的接触 mask。 |

## 示例

```python
from genelab.configs import InteractiveSceneCfg
from genelab.sensor import KinematicContactSensorCfg

scene_cfg = InteractiveSceneCfg(
    sensors=(
        KinematicContactSensorCfg(
            name="fingertip_contacts",
            link_name="right_finger_tip",
            probe_local_pos=(
                (0.0, 0.0, 0.0),
                (0.005, 0.0, 0.0),
                (-0.005, 0.0, 0.0),
            ),
            contact_threshold=2e-4,
        ),
    ),
)
```

## See also

- [ProximitySensor](proximity.md)
- [Concepts — Sensors](../../concepts/sensors.md)
