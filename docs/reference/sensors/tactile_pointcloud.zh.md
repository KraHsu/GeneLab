# PointCloudTactileSensor

刚度 + 剪切点云触觉传感器，底层是 `genesis.sensors.ProximityTaxel`。

在父 link 的网格上采样一组点云（默认 `use_visual_mesh=True`），每帧对一组被
跟踪 link 做接触查询。接触响应由线性 `stiffness` 和可选的 `shear_coupling`
控制 —— 比可变形 elastomer 模型更轻量，但仍能从几何上得到每个探针的触觉
信息。

如果需要带有限刚度的可变形接触动力学，请用
[ElastomerTactileSensor](tactile_elastomer.md)。

## 配置

`PointCloudTactileSensorCfg` 继承 `SensorCfg`。全部字段使用关键字参数。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | `str` | `""` | 继承自 `SensorCfg`，场景内唯一。 |
| `entity_name` | `str` | `"robot"` | 继承自 `SensorCfg`，挂载探针的实体。 |
| `link_name` | `str` | `""` | 探针所在的父 link。必填。 |
| `probe_local_pos` | `tuple[tuple[float, float, float], ...]` | `((0.0, 0.0, 0.0),)` | 探针在父 link 本地坐标系下的位置。 |
| `probe_local_normal` | `tuple[float, float, float]` | `(0.0, 0.0, 1.0)` | 探针前向方向（父 link 本地系）。 |
| `probe_radius` | `float` | `0.01` | 探针球半径（米）。 |
| `track_link_names` | `tuple[str, ...]` | `()` | 非空的接触目标 link 集合。 |
| `n_sample_points` | `int` | `500` | 网格上的点云采样密度。 |
| `use_visual_mesh` | `bool` | `True` | 采样自视觉网格（`True`）或碰撞几何（`False`）。 |
| `stiffness` | `float` | `100.0` | 接触响应的线性刚度（N/m）。 |
| `shear_coupling` | `float` | `0.0` | 切向耦合系数。 |
| `history_length` | `int` | `0` | 透传给 Genesis。`0` 表示只读当前帧；`> 0` 分配 ring buffer。 |

!!! warning "`track_link_names` 必填"
    Genesis 拒绝构造 `track_link_idx` 为空的 `ProximityTaxel`。封装层在最早期
    抛 `ValueError`，务必填入要查询接触的目标 link（例如被操作物体）。

## 数据

Genesis 的 `ProximityTaxel.read()` 返回结构化的
`ProximityTaxelData(force=…, torque=…)`。封装层把两个张量字段提取到
`PointCloudTactileData` 上；`raw` 别名为 `force`，让
`genelab.mdp.rewards.tactile` 中的 shape-agnostic 奖励函数继续可用。

| 字段 | 类型 | 形状 |
|---|---|---|
| `force` | `torch.Tensor` (float32) | `(num_envs, [history,] num_probes, 3)` —— 探针本地系下的接触力向量。 |
| `torque` | `torch.Tensor` (float32) | `(num_envs, [history,] num_probes, 3)` —— 探针本地系下的接触力矩向量。 |
| `raw` | `torch.Tensor` | `force` 的别名。 |

## 示例

```python
from genelab.configs import InteractiveSceneCfg
from genelab.sensor import PointCloudTactileSensorCfg

scene_cfg = InteractiveSceneCfg(
    sensors=(
        PointCloudTactileSensorCfg(
            name="finger_pad",
            link_name="right_finger_tip",
            probe_local_pos=((0.0, 0.0, 0.001),),
            track_link_names=("object",),
            n_sample_points=128,
            stiffness=250.0,
            shear_coupling=0.5,
        ),
    ),
)
```

## 另见

- [ElastomerTactileSensor](tactile_elastomer.md)
- [Concepts — Sensors](../../concepts/sensors.md)
