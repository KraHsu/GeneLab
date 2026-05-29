# ElastomerTactileSensor

Deformable-elastomer point-cloud tactile sensor, backed by
`genesis.sensors.ElastomerTaxel`.

A point cloud is sampled from the parent link's mesh (`use_visual_mesh=True`
by default; `False` falls back to collision geometry) and used as the
contact surface of a deformable elastomer. Per-probe contact response is
governed by the Lamé parameters `lambda_d` (volumetric) and `lambda_s`
(shear). Use this when the task needs a tactile signal that respects
finite-stiffness contact dynamics — peg-in-hole, grasp-stability,
dexterous manipulation.

For tasks where a stiffness-only model is sufficient, the lighter
[PointCloudTactileSensor](tactile_pointcloud.md) is the better fit.

## Configuration

`ElastomerTactileSensorCfg` extends `SensorCfg`. All fields are keyword-only.

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | `""` | Inherited from `SensorCfg`. Unique per scene. |
| `entity_name` | `str` | `"robot"` | Inherited. Which entity carries the elastomer. |
| `link_name` | `str` | `""` | Parent link the elastomer rides on. Required. |
| `probe_local_pos` | `tuple[tuple[float, float, float], ...]` | `((0.0, 0.0, 0.0),)` | Probe positions in the parent link's local frame. |
| `probe_local_normal` | `tuple[float, float, float]` | `(0.0, 0.0, 1.0)` | Probe forward direction in the parent link's local frame. |
| `probe_radius` | `float` | `0.01` | Probe sphere radius (metres). |
| `track_link_names` | `tuple[str, ...]` | `()` | Non-empty set of links the elastomer may deform against. |
| `n_sample_points` | `int` | `500` | Point-cloud density sampled from the mesh. |
| `use_visual_mesh` | `bool` | `True` | Sample from the visual mesh (`True`) or collision geometry (`False`). |
| `lambda_d` | `float` | `700.0` | Volumetric Lamé parameter (elastomer stiffness). |
| `lambda_s` | `float` | `300.0` | Shear Lamé parameter. |
| `history_length` | `int` | `0` | Forwarded to Genesis. `0` = current-step only; `> 0` allocates a ring buffer. |

!!! warning "`track_link_names` is required"
    Genesis rejects an empty `track_link_idx`. The wrapper raises a
    `ValueError` up-front; populate the field with the links the elastomer
    is allowed to deform against (e.g. the manipulated object).

## Data

`ElastomerTactileData.raw` is whatever
`gs.sensors.ElastomerTaxel.read()` returns —
`(num_envs, [history,] ...)` per Genesis's per-sensor cache layout.
[Reward primitives](../../api/reference.md) reduce over every non-batch
axis; consumers needing a specific decomposition (normal vs shear) should
slice it themselves.

## Example

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

## See also

- [PointCloudTactileSensor](tactile_pointcloud.md)
- [Concepts — Sensors](../../concepts/sensors.md)
