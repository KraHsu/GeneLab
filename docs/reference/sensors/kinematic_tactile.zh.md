# KinematicTactileSensor

点 taxel 力 / 力矩传感器，底层是 `genesis.sensors.KinematicTaxel`。

一组点状 taxel 刚性挂在父 link 上，按运动学弹簧-阻尼模型逐探针估计**接触力
与力矩**，模型由 SDF 穿透深度以及探针与所触几何的相对运动驱动。对每个 taxel，
Genesis 计算：

```text
v_n = dot(relative_velocity, probe_normal) * probe_normal
v_t = relative_velocity - v_n
s   = penetration ** normal_exponent
F   = (-normal_stiffness * s * probe_normal) - (normal_damping * s * v_n) - (shear_scalar * v_t)
T   = cross(probe_local_pos, F) - twist_scalar * dot(relative_angular_velocity, probe_normal) * probe_normal
```

接触法向由 `force` 的方向隐式携带 —— 其法向分量与 `probe_local_normal` 对齐。
输出在父 link 的本地坐标系下。

!!! note "保真层级"
    这是几何 / SDF 估计，**不是**求解器的冲量力，也**不是**可变形皮肤 FEM 或
    光学触觉模型。力是穿透与相对运动的运动学弹簧-阻尼读数。若只需布尔接触标志
    或原始穿透深度，见 [KinematicContactSensor](kinematic_contact.md) /
    [KinematicDepthSensor](kinematic_depth.md)。

## 配置

`KinematicTactileSensorCfg` 继承 `SensorCfg`。全部字段使用关键字参数。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | `str` | `""` | 继承自 `SensorCfg`，场景内唯一。 |
| `entity_name` | `str` | `"robot"` | 继承自 `SensorCfg`，挂载 taxel 的实体。 |
| `link_name` | `str` | `""` | taxel 所在的父 link。必填。 |
| `probe_local_pos` | `tuple[tuple[float, float, float], ...]` | `((0.0, 0.0, 0.0),)` | taxel 在父 link 本地坐标系下的位置。 |
| `probe_local_normal` | `tuple[float, float, float]` | `(0.0, 0.0, 1.0)` | link 本地坐标系下共享的接触法向。 |
| `probe_radius` | `float` | `0.01` | SDF 查询半径，单位米。 |
| `probe_radius_noise` | `float` | `0.0` | 双半径 sim2real 扰动幅度，单位米。`0.0` 表示关闭。 |
| `normal_stiffness` | `float` | `1000.0` | 法向力对穿透深度的刚度（N/m）。 |
| `normal_damping` | `float` | `1.0` | 法向力对穿透速率的阻尼。 |
| `normal_exponent` | `float` | `1.0` | 穿透指数。`1.0` 线性，`1.5` Hertzian。必须 `>= 1.0`。 |
| `shear_scalar` | `float` | `1.0` | 切向相对速度对剪切力的系数。 |
| `twist_scalar` | `float` | `1.0` | 相对角速度对扭转力矩的系数。 |
| `history_length` | `int` | `0` | 透传给 Genesis。`0` 表示只读当前帧；`> 0` 分配 ring buffer，`data.force` / `data.torque` 多一个 history 维度。 |

## 数据

| 字段 | 类型 | 形状 |
|---|---|---|
| `force` | `torch.Tensor` (float) | `(num_envs, [history,] num_probes, 3)` —— 父 link 本地坐标系下的弹簧-阻尼力，末轴为 `(tangent_x, tangent_y, normal)`。 |
| `torque` | `torch.Tensor` (float) | `(num_envs, [history,] num_probes, 3)` —— 父 link 本地坐标系下的弹簧-阻尼力矩。 |
| `raw` | `torch.Tensor` (float) | `force` 的别名，供 `genelab.mdp.rewards.tactile` 的奖励原语使用。 |

## 示例

```python
from genelab.configs import InteractiveSceneCfg
from genelab.sensor import KinematicTactileSensorCfg

scene_cfg = InteractiveSceneCfg(
    sensors=(
        KinematicTactileSensorCfg(
            name="fingertip_taxels",
            link_name="right_finger_tip",
            probe_local_pos=(
                (0.0, 0.0, 0.0),
                (0.005, 0.0, 0.0),
                (-0.005, 0.0, 0.0),
            ),
            probe_local_normal=(0.0, 0.0, 1.0),
            normal_stiffness=1000.0,
        ),
    ),
)
```

指尖压到方块上时，离开几何读数 `≈ 0`，接触时产生沿法向的力（例如
`normal_stiffness=1000` 在 `0.01` m 穿透下 `force ≈ [0, 0, 10.0]` N，因为
`F_n = normal_stiffness · penetration`）。

## 另见

- [KinematicDepthSensor](kinematic_depth.md)
- [PointCloudTactileSensor](tactile_pointcloud.md)
- [Concepts — Sensors](../../concepts/sensors.md)
