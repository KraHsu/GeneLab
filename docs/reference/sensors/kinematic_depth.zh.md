# KinematicDepthSensor

点接触深度探针，底层是 `genesis.sensors.ContactDepthProbe`。

一组点状探针刚性挂在父 link 上，每帧报告对周围几何的**穿透深度**（米），
由每个探针周围的 SDF 查询得到，而非物理求解器的接触冲量。相比
[KinematicContactSensor](kinematic_contact.md)（`ContactProbe`）返回 Genesis
内部阈值化后的 `bool` mask，本传感器返回底层的连续深度：非负，探针离开几何
时为 `0`，随穿透加深而增大。

!!! note "保真层级"
    这是几何 / SDF 触觉 —— 只有穿透深度，没有剪切或形变，**不是**可变形皮肤
    FEM 或光学触觉模型。`probe_radius_noise > 0` 启用 Genesis 的双半径 sim2real
    方案，在 measured 分支上逐帧扰动每个探针半径。

## 配置

`KinematicDepthSensorCfg` 继承 `SensorCfg`。全部字段使用关键字参数。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | `str` | `""` | 继承自 `SensorCfg`，场景内唯一。 |
| `entity_name` | `str` | `"robot"` | 继承自 `SensorCfg`，挂载探针的实体。 |
| `link_name` | `str` | `""` | 探针所在的父 link。必填。 |
| `probe_local_pos` | `tuple[tuple[float, float, float], ...]` | `((0.0, 0.0, 0.0),)` | 探针在父 link 本地坐标系下的位置。 |
| `probe_radius` | `float` | `0.01` | SDF 查询半径，单位米。 |
| `probe_radius_noise` | `float` | `0.0` | 双半径 sim2real 扰动幅度，单位米。`0.0` 表示关闭。 |
| `history_length` | `int` | `0` | 透传给 Genesis。`0` 表示只读当前帧；`> 0` 分配 ring buffer，`data.depth` 多一个 history 维度。 |

## 数据

| 字段 | 类型 | 形状 |
|---|---|---|
| `depth` | `torch.Tensor` (float) | `(num_envs, [history,] num_probes)` —— 穿透深度，单位米，非负。 |
| `raw` | `torch.Tensor` (float) | `depth` 的别名，供 `genelab.mdp.rewards.tactile` 的奖励原语使用。 |

## 示例

```python
from genelab.configs import InteractiveSceneCfg
from genelab.sensor import KinematicDepthSensorCfg

scene_cfg = InteractiveSceneCfg(
    sensors=(
        KinematicDepthSensorCfg(
            name="fingertip_depth",
            link_name="right_finger_tip",
            probe_local_pos=(
                (0.0, 0.0, 0.0),
                (0.005, 0.0, 0.0),
                (-0.005, 0.0, 0.0),
            ),
            probe_radius=0.01,
        ),
    ),
)
```

指尖压到方块上时，离开几何读数为 `0`，完全穿透时增大到约一个探针半径
（例如 `0.01` m 的探针压在表面上读出 `depth ≈ 0.010` m）。

## 另见

- [KinematicContactSensor](kinematic_contact.md)
- [KinematicTactileSensor](kinematic_tactile.md)
- [Concepts — Sensors](../../concepts/sensors.md)
