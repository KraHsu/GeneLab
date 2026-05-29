# TemperatureGridSensor

Coarse thermal-grid sensor, backed by `genesis.sensors.TemperatureGrid`.

A regular `(nx, ny, nz)` grid of temperature samples carried on a robot
link. Useful as a low-cost thermal proxy for tasks where contact-driven heat
matters — friction-warmed end-effectors, exhaust plumes, thermal-aware
grasping. This is the minimum-viable wrapper exposing grid resolution and
ambient temperature; the richer Genesis surface (per-material
`properties_dict`, `heat_generation`, `convection_coefficient`) is omitted
from this revision.

## Configuration

`TemperatureGridSensorCfg` extends `SensorCfg`. All fields are keyword-only.

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | `""` | Inherited from `SensorCfg`. Unique per scene. |
| `entity_name` | `str` | `"robot"` | Inherited. Which entity carries the grid. |
| `link_name` | `str` | `""` | Link that carries the grid. Required. |
| `grid_size` | `tuple[int, int, int]` | `(1, 1, 1)` | `(nx, ny, nz)` resolution in the link's local frame. |
| `ambient_temperature` | `float \| None` | `None` | Seed temperature in degrees Celsius. `None` falls through to Genesis's per-material defaults. |
| `history_length` | `int` | `0` | Forwarded to Genesis. `0` = current-step only; `> 0` allocates a ring buffer and `data.temperature` carries a leading history axis. |

## Data

`TemperatureGridData.temperature` is
`(num_envs, [history,] nx, ny, nz)` in degrees Celsius.

## Example

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

## See also

- [KinematicContactSensor](kinematic_contact.md)
- [Concepts — Sensors](../../concepts/sensors.md)
