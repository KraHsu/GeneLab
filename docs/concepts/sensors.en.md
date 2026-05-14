# Sensors

Genesis does not parse the MJCF `<sensor>` block, so GeneLab introduces a backend-agnostic
sensor abstraction. The interface mirrors mjlab's `SensorCfg` / `Sensor[T]`, but every concrete
sensor reads from the env's `RobotState` instead of MuJoCo sensordata — observation and reward
terms transfer between backends without ceremony.

## Lifecycle

`bind(env)` runs once at construction. Each control step, `update(dt)` invalidates the cache;
the first access to `data` triggers `_compute_data` lazily. `reset(env_ids)` invalidates the
cache and lets stateful sensors clear per-env buffers.

```python
class Sensor[T](ABC):
    def bind(self, env: "ManagerBasedRlEnv") -> None: ...
    @property
    def data(self) -> T: ...
    def update(self, dt: float) -> None: ...
    def reset(self, env_ids: torch.Tensor | None = None) -> None: ...
    @abstractmethod
    def _compute_data(self) -> T: ...
```

The env wires the lifecycle automatically: sensors built from `InteractiveSceneCfg.sensors`
get `update` called after each articulation refresh and `reset` called from inside
`_reset_idx`, so reward and observation terms always see fresh sensor data.

## Registering on a scene

`InteractiveSceneCfg.sensors` is a tuple of `SensorCfg`. `ManagerBasedRlEnv.__init__` calls
`build()` on each cfg and binds the resulting sensor to the env. Access at runtime is via
`env.sensors[name].data`.

```python
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.sensor import BodyVelocitySensorCfg, ContactSensorCfg

simulation = SimulationCfg(num_envs=4096)
scene = InteractiveSceneCfg(
    sensors=(
        BodyVelocitySensorCfg(
            name="imu_lin_vel",
            link_name="pelvis",
            offset=(0.04525, 0.0, -0.08339),
            measure="lin_vel",
        ),
        ContactSensorCfg(
            name="feet_ground_contact",
            link_names=("left_ankle_roll_link", "right_ankle_roll_link"),
            track_air_time=True,
        ),
    ),
)
```

## Built-in sensors

### BodyVelocitySensor

Mirrors a MuJoCo `<velocimeter>` / `<gyro>` on a site rigidly attached to a robot link. Returns
the linear or angular velocity of the site in the link's body frame, with a configurable
lever-arm offset and optional per-env uniform bias.

| Field | Type | Meaning |
|-------|------|---------|
| `link_name` | `str` | Link the site is rigidly attached to. |
| `offset` | `tuple[float, float, float]` | Site position in the link's local frame. Ignored for `ang_vel`. |
| `measure` | `Literal["lin_vel", "ang_vel"]` | Velocimeter (linear) or gyro (angular). |
| `bias_range` | `tuple[float, float] \| None` | Uniform per-env bias resampled on `reset`. |

The velocimeter math is `v_site = v_link + ω × (R_link · offset)` rotated into the body frame —
matching MuJoCo's lever-arm convention so a GeneLab-trained policy reads the same signal as the
mjlab reference.

### IMUSensor

Inertial Measurement Unit at a site rigidly attached to a link. Outputs orientation
(`link_quat_w`), body-frame projected unit gravity, and body-frame linear / angular
acceleration. Accelerations are computed by finite difference of the world-frame velocity
buffers; the first control step after every `reset` returns zero acceleration to avoid a
spurious spike from a stale `prev` value — document this in any reward / observation term
that consumes `lin_acc_b` or `ang_acc_b` near reset boundaries.

| Field | Type | Meaning |
|-------|------|---------|
| `link_name` | `str` | Link the IMU is rigidly attached to. |
| `offset` | `tuple[float, float, float]` | Site position in the link's local frame; contributes the lever-arm term to linear acceleration. |
| `gravity_bias` | `bool` | When `True`, output is the specific force `R^T (a_w - g_w)` — matches a real accelerometer at rest reading `+g` along its up axis. |
| `bias_range_lin_acc` | `tuple[float, float] \| None` | Per-env constant accelerometer bias resampled on `reset`. |
| `bias_range_ang_acc` | `tuple[float, float] \| None` | Per-env constant gyro-derivative bias resampled on `reset`. |

`IMUSensor` does not output linear or angular velocity — that remains the role of
`BodyVelocitySensor`, which can coexist on the same link when both signals are needed. The
finite-difference at the control rate is **not** filtered internally; layer
`ObservationTermCfg.noise` + `scale` on top, or apply a custom EMA in the observation term.

### ContactSensor

Per-link aggregate of `robot.get_links_net_contact_force()`. With `track_air_time=True`, an
internal state machine advances `current_air_time` / `current_contact_time` per env, and snaps
the completed durations into `last_air_time` / `last_contact_time` at the contact transition.

| Field | Type | Meaning |
|-------|------|---------|
| `link_names` | `tuple[str, ...]` | Explicit list of link names to monitor. |
| `link_names_expr` | `str \| None` | Regex matched against `env.link_names`. |
| `force_threshold` | `float` | Force magnitude (N) above which `found` is true. |
| `track_air_time` | `bool` | Allocate the air-time / contact-time state machine. |

`data` is a `ContactData` dataclass with `force`, `force_norm`, `found`, plus the four air-time
buffers. The matching obs terms — `mdp.foot_air_time`, `mdp.foot_contact`,
`mdp.foot_contact_forces` — read straight off this dataclass.

### Ray-cast patterns

`RayCastSensorCfg.pattern` accepts any of the three bundled pattern dataclasses. Custom
patterns satisfy the same informal protocol — `num_rays() -> int` and
`generate(device) -> (starts, dirs)` with both tensors shaped `(M, 3)` in the sensor's local
frame — and slot in without changing `RayCastSensor` itself.

`GridPattern` is the default; rays are parallel and arranged on a 2D rectangle. `RingPattern`
emits `num_horizontal × num_vertical` rays from the origin, evenly spaced in azimuth and
elevation — the typical multi-line LIDAR layout. `HemispherePattern` distributes
`num_rays_target` rays on a spherical cap of half-angle `polar_fov_deg` around `pole_axis`
using a Fibonacci lattice; 90° covers a full hemisphere, 180° a full sphere.

| Pattern | Key fields | Use case |
|---------|------------|----------|
| `GridPattern` | `resolution`, `size`, `direction` | Height-scan grids, area sweeps |
| `RingPattern` | `num_horizontal`, `num_vertical`, `horizontal_fov_deg`, `vertical_fov_deg` | Planar / multi-line LIDAR |
| `HemispherePattern` | `num_rays_target`, `pole_axis`, `polar_fov_deg` | Proximity dome, downward coverage |

`RingPattern` treats a horizontal span of exactly ±360° as a wrap-around and drops the
duplicate closing azimuth; any other span (e.g. `(-30, 30)` for a forward-facing scanner) is
inclusive on both endpoints. `HemispherePattern.num_rays()` returns `num_rays_target`
exactly — the Fibonacci lattice produces no rounding error.

### TerrainHeightSensor

2D grid of downward rays anchored to a robot link. Output is per-ray height above the terrain
(positive = above), useful as a privileged `height_scan` critic observation. The default
backend intersects every ray against a horizontal plane at `ground_height`; when the scene
attaches a `TerrainImporter`, the inner `RayCastSensor` bilinearly samples the height-field
instead. Subclassing `RayCastSensor` and overriding `_intersect_world_rays` is the extension
point for BVH or other custom backends.

| Field | Type | Meaning |
|-------|------|---------|
| `link_name` | `str` | Anchor link for the grid origin. |
| `pattern` | `GridPattern \| RingPattern \| HemispherePattern` | Pattern geometry. |
| `attach_yaw_only` | `bool` | Rotate the pattern by yaw only so it stays horizon-aligned. |
| `max_distance` | `float` | Distance clamp for the ray cast. |
| `ground_height` | `float` | Plane height used by the default flat-plane backend. |

## Observations with noise

`ObservationTermCfg.noise` accepts an additive noise model (`Unoise(n_min, n_max)` or
`Gnoise(mean, std)`). `ObservationGroupCfg.enable_corruption` gates whether the noise is
applied — disabled by default to keep the critic on ground truth. The per-term pipeline order
is **noise → scale → clip**, so noise magnitudes live in raw signal space: `Unoise(-1.5, 1.5)`
on a raw `joint_vel` term with `scale=0.05` ends up as ±0.075 final jitter.

The canonical pattern shares terms between policy and critic and differs only on
`enable_corruption`:

```python
from genelab import mdp
from genelab.managers import ObservationGroupCfg, ObservationTermCfg
from genelab.mdp.noise import Unoise


def _obs_terms() -> dict[str, ObservationTermCfg]:
    return {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.sensor_data,
            params={"sensor_name": "imu_lin_vel"},
            noise=Unoise(-0.5, 0.5),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(-1.5, 1.5),
        ),
    }


policy = ObservationGroupCfg(enable_corruption=True, terms=_obs_terms())
critic = ObservationGroupCfg(enable_corruption=False, terms=_obs_terms())
```

## Writing a custom sensor

Subclass `Sensor[T]` with the desired return type and implement `_compute_data`. Override
`bind` to cache link indices once, `update` to advance integrators, and `reset` to clear per-env
state — always calling `super()` first so the cache-invalidation chain stays intact.

```python
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class JointTorqueSensorCfg(SensorCfg):
    def build(self) -> "JointTorqueSensor":
        return JointTorqueSensor(self)


class JointTorqueSensor(Sensor[torch.Tensor]):
    def bind(self, env: "ManagerBasedRlEnv") -> None:
        super().bind(env)
        # Cache anything that depends on env.link_names / env.joint_names here.

    def _compute_data(self) -> torch.Tensor:
        assert self._env is not None
        rs = self._env.robot_state
        return self._env.joint_kp * (rs.joint_pos - self._env.default_joint_pos)
```

Add the cfg to `InteractiveSceneCfg.sensors` and the sensor is reachable as
`env.sensors[name].data`.

## See also

- [Configs](configs.md)
- [API Reference](../api/reference.md)
