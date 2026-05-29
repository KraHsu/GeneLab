# PointCloudTactileSensor

Stiffness-and-shear point-cloud tactile sensor, backed by
`genesis.sensors.ProximityTaxel`.

A point cloud is sampled from the parent link's mesh (`use_visual_mesh=True`
by default) and queried against a set of tracked links each step. The
contact response is governed by a linear `stiffness` and an optional
`shear_coupling` — lighter than the deformable elastomer model but still
gives per-probe tactile information sampled from the geometry.

For tasks that need finite-stiffness deformable contact dynamics, see
[ElastomerTactileSensor](tactile_elastomer.md).

## Configuration

`PointCloudTactileSensorCfg` extends `SensorCfg`. All fields are keyword-only.

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | `""` | Inherited from `SensorCfg`. Unique per scene. |
| `entity_name` | `str` | `"robot"` | Inherited. Which entity carries the probe. |
| `link_name` | `str` | `""` | Parent link the probe rides on. Required. |
| `probe_local_pos` | `tuple[tuple[float, float, float], ...]` | `((0.0, 0.0, 0.0),)` | Probe positions in the parent link's local frame. |
| `probe_local_normal` | `tuple[float, float, float]` | `(0.0, 0.0, 1.0)` | Probe forward direction in the parent link's local frame. |
| `probe_radius` | `float` | `0.01` | Probe sphere radius (metres). |
| `track_link_names` | `tuple[str, ...]` | `()` | Non-empty set of links to sense contact against. |
| `n_sample_points` | `int` | `500` | Point-cloud density sampled from the mesh. |
| `use_visual_mesh` | `bool` | `True` | Sample from the visual mesh (`True`) or collision geometry (`False`). |
| `stiffness` | `float` | `100.0` | Linear stiffness for the contact response (N/m). |
| `shear_coupling` | `float` | `0.0` | Tangential coupling coefficient. |
| `history_length` | `int` | `0` | Forwarded to Genesis. `0` = current-step only; `> 0` allocates a ring buffer. |

!!! warning "`track_link_names` is required"
    Genesis rejects an empty `track_link_idx`. The wrapper raises a
    `ValueError` up-front; populate it with the links to sense against
    (e.g. the manipulated object).

## Data

`PointCloudTactileData.raw` is whatever
`gs.sensors.ProximityTaxel.read()` returns —
`(num_envs, [history,] ...)` per Genesis's per-sensor cache layout. The
reward primitives in `genelab.mdp.rewards.tactile` reduce over every
non-batch axis; consumers that need a specific decomposition should
slice it themselves.

## Example

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

## See also

- [ElastomerTactileSensor](tactile_elastomer.md)
- [Concepts — Sensors](../../concepts/sensors.md)
