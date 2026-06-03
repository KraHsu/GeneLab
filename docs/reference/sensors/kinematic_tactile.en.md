# KinematicTactileSensor

Point taxel force/torque sensor, backed by `genesis.sensors.KinematicTaxel`.

A set of point taxels ride on a parent link and estimate **contact force and
torque** per probe from a kinematic spring-damper model driven by SDF
penetration depth and the relative motion of the probe and the geometry it
touches. For each taxel Genesis evaluates:

```text
v_n = dot(relative_velocity, probe_normal) * probe_normal
v_t = relative_velocity - v_n
s   = penetration ** normal_exponent
F   = (-normal_stiffness * s * probe_normal) - (normal_damping * s * v_n) - (shear_scalar * v_t)
T   = cross(probe_local_pos, F) - twist_scalar * dot(relative_angular_velocity, probe_normal) * probe_normal
```

The contact normal is carried implicitly by the direction of `force` — its
normal component aligns with `probe_local_normal`. Outputs are in the parent
link's local frame.

!!! note "Fidelity"
    This is a geometric / SDF estimate, **not** the solver's impulse force and
    **not** a deformable-skin FEM or optical tactile model. Force is a kinematic
    spring-damper readout of penetration and relative motion. For a binary
    contact flag or raw penetration depth instead, see
    [KinematicContactSensor](kinematic_contact.md) /
    [KinematicDepthSensor](kinematic_depth.md).

## Configuration

`KinematicTactileSensorCfg` extends `SensorCfg`. All fields are keyword-only.

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | `""` | Inherited from `SensorCfg`. Unique per scene. |
| `entity_name` | `str` | `"robot"` | Inherited. Which entity carries the taxels. |
| `link_name` | `str` | `""` | Parent link the taxels ride on. Required. |
| `probe_local_pos` | `tuple[tuple[float, float, float], ...]` | `((0.0, 0.0, 0.0),)` | Taxel positions in the parent link's local frame. |
| `probe_local_normal` | `tuple[float, float, float]` | `(0.0, 0.0, 1.0)` | Shared contact normal in the link's local frame. |
| `probe_radius` | `float` | `0.01` | SDF query radius in metres. |
| `probe_radius_noise` | `float` | `0.0` | Dual-radius sim2real perturbation magnitude in metres. `0.0` disables it. |
| `normal_stiffness` | `float` | `1000.0` | Normal-force stiffness against penetration depth (N/m). |
| `normal_damping` | `float` | `1.0` | Normal-force damping against penetration-rate. |
| `normal_exponent` | `float` | `1.0` | Penetration exponent. `1.0` linear, `1.5` Hertzian. Must be `>= 1.0`. |
| `shear_scalar` | `float` | `1.0` | Coefficient on tangential relative velocity for shear force. |
| `twist_scalar` | `float` | `1.0` | Coefficient on relative angular velocity for twist torque. |
| `history_length` | `int` | `0` | Forwarded to Genesis. `0` = current-step only; `> 0` allocates a ring buffer carried as a leading axis on `data.force` / `data.torque`. |

## Data

| Field | Type | Shape |
|---|---|---|
| `force` | `torch.Tensor` (float) | `(num_envs, [history,] num_probes, 3)` — spring-damper force in the parent link's local frame, last axis `(tangent_x, tangent_y, normal)`. |
| `torque` | `torch.Tensor` (float) | `(num_envs, [history,] num_probes, 3)` — spring-damper torque in the parent link's local frame. |
| `raw` | `torch.Tensor` (float) | Alias of `force`, for the reward primitives in `genelab.mdp.rewards.tactile`. |

## Example

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

A fingertip pressed onto a box reads `≈ 0` while clear, and a normal-aligned
force on contact (e.g. `force ≈ [0, 0, 10.0]` N for `normal_stiffness=1000` at
`0.01` m penetration, since `F_n = normal_stiffness · penetration`).

## See also

- [KinematicDepthSensor](kinematic_depth.md)
- [PointCloudTactileSensor](tactile_pointcloud.md)
- [Concepts — Sensors](../../concepts/sensors.md)
