# KinematicDepthSensor

Point contact-depth probe, backed by `genesis.sensors.ContactDepthProbe`.

A set of point probes ride on a parent link and report **penetration depth**
(metres) against surrounding geometry, computed from SDF queries around each
probe rather than the physics solver's contact impulses. Where
[KinematicContactSensor](kinematic_contact.md) (`ContactProbe`) returns a
Genesis-thresholded `bool` mask, this returns the underlying continuous depth:
non-negative, `0` when the probe is clear of geometry and growing with
inter-penetration.

!!! note "Fidelity"
    This is geometric / SDF tactile — penetration depth only, no shear or
    deformation. It is **not** a deformable-skin FEM or optical tactile model.
    `probe_radius_noise > 0` enables Genesis's dual-radius sim2real scheme, which
    perturbs each probe radius per step on the measured branch.

## Configuration

`KinematicDepthSensorCfg` extends `SensorCfg`. All fields are keyword-only.

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | `""` | Inherited from `SensorCfg`. Unique per scene. |
| `entity_name` | `str` | `"robot"` | Inherited. Which entity carries the probes. |
| `link_name` | `str` | `""` | Parent link the probes ride on. Required. |
| `probe_local_pos` | `tuple[tuple[float, float, float], ...]` | `((0.0, 0.0, 0.0),)` | Probe positions in the parent link's local frame. |
| `probe_radius` | `float` | `0.01` | SDF query radius in metres. |
| `probe_radius_noise` | `float` | `0.0` | Dual-radius sim2real perturbation magnitude in metres. `0.0` disables it. |
| `history_length` | `int` | `0` | Forwarded to Genesis. `0` = current-step only; `> 0` allocates a ring buffer carried as a leading axis on `data.depth`. |

## Data

| Field | Type | Shape |
|---|---|---|
| `depth` | `torch.Tensor` (float) | `(num_envs, [history,] num_probes)` — penetration depth in metres, non-negative. |
| `raw` | `torch.Tensor` (float) | Alias of `depth`, for the reward primitives in `genelab.mdp.rewards.tactile`. |

## Example

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

A fingertip pressed onto a box reads `0` while clear and grows to roughly the
probe radius at full inter-penetration (e.g. `depth ≈ 0.010` m for a `0.01` m
probe resting on a surface).

## See also

- [KinematicContactSensor](kinematic_contact.md)
- [KinematicTactileSensor](kinematic_tactile.md)
- [Concepts — Sensors](../../concepts/sensors.md)
