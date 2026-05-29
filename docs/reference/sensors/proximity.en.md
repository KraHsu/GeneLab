# ProximitySensor

Surface-distance proximity probe, backed by `genesis.sensors.SurfaceDistanceProbe`.

A set of point probes ride on a parent link and report the distance to the
nearest surface among a configurable set of *tracked* links. Useful for
proximity gating, soft self-collision avoidance, and time-to-contact
estimation when terrain or another link is the relevant target.

## Configuration

`ProximitySensorCfg` extends `SensorCfg`. All fields are keyword-only.

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | `""` | Inherited from `SensorCfg`. Must be unique on a scene. |
| `entity_name` | `str` | `"robot"` | Inherited. Which entity carries the probes. |
| `link_name` | `str` | `""` | Parent link the probes ride on. Required (must be in `entity.link_names`). |
| `probe_local_pos` | `tuple[tuple[float, float, float], ...]` | `((0.0, 0.0, 0.0),)` | Probe positions in the parent link's local frame. |
| `probe_radius` | `float` | `10.0` | Per-probe search radius; also the saturation value reported when no tracked surface lies within range. |
| `track_link_names` | `tuple[str, ...]` | `()` | Non-empty set of links to measure distance against. Resolved against the same entity's `link_names`. |
| `history_length` | `int` | `0` | Forwarded to Genesis. `0` = current-step only; `> 0` allocates a ring buffer and `data.distance` carries a leading history axis. |

!!! warning "`track_link_names` is required"
    Genesis refuses to construct a `SurfaceDistanceProbe` with an empty
    `track_link_idx`. The wrapper raises a `ValueError` up-front with the
    same message; populate the field with the targets to measure against.

## Data

`ProximityData.distance` is `(num_envs, [history,] num_probes)`, in metres,
clipped to `cfg.probe_radius` when no tracked surface lies within reach.

## Example

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
