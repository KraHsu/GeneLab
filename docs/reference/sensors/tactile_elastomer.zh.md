# ElastomerTactileSensor

可变形 elastomer 点云触觉传感器，底层是
`genesis.sensors.ElastomerTaxel`。

在父 link 的网格上采样一组点云（默认 `use_visual_mesh=True`，关掉则使用碰撞
几何），作为可变形 elastomer 的接触表面。每个探针的接触响应由 Lamé 参数
`lambda_d`（体积）和 `lambda_s`（剪切）控制。适用于需要保留有限刚度接触动力学
的触觉任务 —— peg-in-hole、抓取稳定性、灵巧操作。

如果只需要 stiffness-only 模型，更轻量的
[PointCloudTactileSensor](tactile_pointcloud.md) 更合适。

## 配置

`ElastomerTactileSensorCfg` 继承 `SensorCfg`。全部字段使用关键字参数。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | `str` | `""` | 继承自 `SensorCfg`，场景内唯一。 |
| `entity_name` | `str` | `"robot"` | 继承自 `SensorCfg`，挂载 elastomer 的实体。 |
| `link_name` | `str` | `""` | Elastomer 所在的父 link。必填。 |
| `probe_local_pos` | `tuple[tuple[float, float, float], ...]` | `((0.0, 0.0, 0.0),)` | 探针在父 link 本地坐标系下的位置。 |
| `probe_local_normal` | `tuple[float, float, float]` | `(0.0, 0.0, 1.0)` | 探针前向方向（父 link 本地系）。 |
| `probe_radius` | `float` | `0.01` | 探针球半径（米）。 |
| `track_link_names` | `tuple[str, ...]` | `()` | 非空的 elastomer 可变形目标 link 集合。 |
| `n_sample_points` | `int` | `500` | 网格上的点云采样密度。 |
| `use_visual_mesh` | `bool` | `True` | 采样自视觉网格（`True`）或碰撞几何（`False`）。 |
| `lambda_d` | `float` | `700.0` | 体积 Lamé 参数（elastomer 刚度）。 |
| `lambda_s` | `float` | `300.0` | 剪切 Lamé 参数。 |
| `history_length` | `int` | `0` | 透传给 Genesis。`0` 表示只读当前帧；`> 0` 分配 ring buffer。 |

!!! warning "`track_link_names` 必填"
    Genesis 拒绝构造 `track_link_idx` 为空的 `ElastomerTaxel`。封装层在最早期
    抛 `ValueError`，务必填入 elastomer 可变形的目标 link（例如被操作物体）。

## 数据

`ElastomerTactileData.raw` 是 `gs.sensors.ElastomerTaxel.read()` 的原始返回
值 —— `(num_envs, [history,] ...)`，对应 Genesis 内部 per-sensor 缓存形状。
[奖励函数](../../api/reference.md) 在 batch 之外的所有轴上归并；需要分解
法向 / 切向的消费者请自行切片。

## 示例

```python
from genelab.configs import InteractiveSceneCfg
from genelab.sensor import ElastomerTactileSensorCfg

scene_cfg = InteractiveSceneCfg(
    sensors=(
        ElastomerTactileSensorCfg(
            name="finger_pad",
            link_name="right_finger_tip",
            probe_local_pos=((0.0, 0.0, 0.005),),
            track_link_names=("peg",),
            n_sample_points=256,
            lambda_d=600.0,
            lambda_s=250.0,
        ),
    ),
)
```

## 另见

- [PointCloudTactileSensor](tactile_pointcloud.md)
- [Concepts — Sensors](../../concepts/sensors.md)
