# KinematicContactSensor

Point contact probe, backed by `genesis.sensors.ContactProbe`.

A set of point probes ride on a parent link and report binary contact.
Genesis thresholds internally at `cfg.contact_threshold` (default `1e-4` m
of inter-penetration) and returns the pre-thresholded boolean mask. Use
this where the question is "did this point on the robot touch something"
without invoking the deformable-elastomer tactile stack.

## Configuration

`KinematicContactSensorCfg` extends `SensorCfg`. All fields are keyword-only.

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | `""` | Inherited from `SensorCfg`. Unique per scene. |
| `entity_name` | `str` | `"robot"` | Inherited. Which entity carries the probes. |
| `link_name` | `str` | `""` | Parent link the probes ride on. Required. |
| `probe_local_pos` | `tuple[tuple[float, float, float], ...]` | `((0.0, 0.0, 0.0),)` | Probe positions in the parent link's local frame. |
| `probe_radius` | `float` | `0.01` | Probe sphere radius in metres (Genesis default). |
| `contact_threshold` | `float` | `1e-4` | Depth threshold (metres) used both by Genesis to mark contact and by the wrapper to derive `data.in_contact`. |
| `history_length` | `int` | `0` | Forwarded to Genesis. `0` = current-step only; `> 0` allocates a ring buffer carried as a leading axis on `data.depth` / `data.in_contact`. |

## Data

| Field | Type | Shape |
|---|---|---|
| `in_contact` | `torch.Tensor` (bool) | `(num_envs, [history,] num_probes)` — Genesis-thresholded contact mask. |

## Example

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
