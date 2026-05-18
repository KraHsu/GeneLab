# MDP terms reference

`genelab.mdp` is a library of plain Python callables that plug straight into
the seven managers. Every function follows the same signature shape — `(env,
**params) -> torch.Tensor` for observation / reward / termination, `(env,
env_ids, **params) -> None` for events, `(env, env_ids, **params) -> Tensor`
for curriculums — and is referenced by the corresponding `*TermCfg.func` (or
`class_type`) field.

## Reusable building blocks

Every name listed below is re-exported by `genelab.mdp.__init__`, so
`from genelab import mdp` is enough to reach them. The full module map:

```
genelab.mdp.actions       → JointPositionAction(Cfg)
genelab.mdp.commands      → UniformVelocityCommand(Cfg), MotionCommand(Cfg), MotionLoader
genelab.mdp.observations  → 16 observation functions
genelab.mdp.rewards       → 15 reward functions / classes
genelab.mdp.terminations  → 6 termination functions
genelab.mdp.events        → 3 event functions
genelab.mdp.curriculums   → 1 curriculum function
genelab.mdp.noise         → NoiseCfg base + Unoise + Gnoise
```

The output-shape column below uses `B = num_envs`, `D = action / joint /
sensor dimension`, `N = number of bodies or feet`. Reward and termination
terms always return shape `(B,)`; observation terms return `(B, D)` (or are
auto-unsqueezed from `(B,)` by `ObservationManager`).

## Actions

| Name | Signature (Cfg fields used) | Behavior |
|---|---|---|
| `JointPositionAction` | `JointPositionActionCfg(joint_names, scale, use_default_offset, asset_name)` | `target = default + scale * raw_action`. Targets are dispatched via `Articulation.write_joint_targets_partial`, which routes each joint to its actuator group's declared channel (position for implicit PD, force for `IdealPDActuator` / `DCMotorActuator`). `scale=None` (default) inherits per-joint scale from `Articulation.action_scale_tensor`; a `float` or `dict[str, float]` overrides it. Patterns are regex; unmatched patterns raise. |

```python
from genelab.mdp.actions.joint_position import JointPositionActionCfg

actions_cfg = {
    "panda_arm": JointPositionActionCfg(
        asset_name="robot",
        joint_names=(r"^joint[1-7]$",),
        use_default_offset=True,
    ),
}
```

## Commands

| Name | Signature (Cfg fields used) | Behavior |
|---|---|---|
| `UniformVelocityCommand` | `UniformVelocityCommandCfg(ranges, rel_standing_envs, heading_command, heading_control_stiffness, resampling_time_range, asset_name)` | Body-frame `[lin_vel_x, lin_vel_y, ang_vel_z]` per env, resampled uniformly from `ranges` every `resampling_time_range` seconds. With `heading_command=True` (default), `ang_vel_z` is rewritten each step to drive the body toward a sampled heading. `rel_standing_envs` zeroes a fraction of resampled commands so the policy also sees a "stand still" target. `command` shape `(B, 3)`. |
| `MotionCommand` | `MotionCommandCfg(motion_file, anchor_body_name, body_names, motion_body_order, motion_joint_order, pose_range, velocity_range, sampling_mode, ...)` | Drives the env toward a recorded motion clip frame-by-frame. Reads the NPZ schema produced by mjlab's `csv_to_npz`. `motion_body_order` / `motion_joint_order` describe the NPZ's native body / joint axis ordering (typically mjlab's MJCF DFS) so `MotionLoader` can permute the data into the runtime robot's order — required when the source uses a different traversal than Genesis. Exposes anchor + multi-body reference and current pose / velocity for the relative-pose reward functions in the next section. `sampling_mode="start"` always begins at frame 0; `"uniform"` (default) samples a random frame. |
| `MotionLoader` | `MotionLoader(motion_file, body_indexes, device, joint_perm=None)` | Helper class consumed by `MotionCommand`. Loads the NPZ into device tensors, applies the optional `joint_perm` to the joint axis, and slices `body_pos_w / body_quat_w / body_lin_vel_w / body_ang_vel_w` down to the bodies the command tracks. Direct use is rare — instantiate `MotionCommandCfg` instead. |

```python
from genelab.mdp.commands.velocity_command import UniformVelocityCommandCfg

commands_cfg = {
    "base_velocity": UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0),
        ),
        heading_command=True,
    ),
}
```

## Observations

All listed functions take `env` plus the noted kwargs and return a tensor of
shape `(B, D)` or `(B,)` (auto-unsqueezed to `(B, 1)` by the manager).

| Name | Params | Shape | Source |
|---|---|---|---|
| `base_lin_vel` | — | `(B, 3)` | Body-frame floating-base linear velocity. |
| `base_ang_vel` | — | `(B, 3)` | Body-frame floating-base angular velocity. |
| `projected_gravity` | — | `(B, 3)` | World gravity projected into the body frame (IMU-orientation proxy). |
| `joint_pos_rel` | — | `(B, num_dofs)` | Joint positions minus the default pose. |
| `joint_vel_rel` | — | `(B, num_dofs)` | Joint velocities (default is zero — same as raw `joint_vel`). |
| `last_action` | — | `(B, total_action_dim)` | The action tensor the manager processed last step. |
| `generated_commands` | `command_name` | `(B, command_dim)` | `env.command_manager.get_command(command_name)`. |
| `sensor_data` | `sensor_name` | varies | Per-step cached tensor on `env.sensors[sensor_name]`. Use this for IMU, FrameTransformer, ray-cast outputs that are already tensors. |
| `foot_air_time` | `sensor_name` (must be a `ContactSensor`) | `(B, N_feet)` | Current per-foot air time (zero while in contact). |
| `foot_contact` | `sensor_name` (must be a `ContactSensor`) | `(B, N_feet)` | Binary contact mask as float. |
| `foot_contact_forces` | `sensor_name` (must be a `ContactSensor`) | `(B, N_feet * 3)` | Per-foot contact force, compressed via `sign(f) * log1p(|f|)` and flattened. |
| `height_scan` | `sensor_name` (must be a `TerrainHeightSensor`) | `(B, num_rays)` | Per-ray heights above the terrain. |
| `motion_anchor_pos_b` | `command_name` (must be a `MotionCommand`) | `(B, 3)` | Reference anchor position expressed in the robot's anchor frame. |
| `motion_anchor_ori_b` | `command_name` | `(B, 6)` | Reference 6D orientation (first two columns of the rotation matrix). |
| `robot_body_pos_b` | `command_name` | `(B, N_bodies * 3)` | Per-body positions in the robot's anchor frame (privileged critic obs). |
| `robot_body_ori_b` | `command_name` | `(B, N_bodies * 6)` | Per-body 6D orientations in the robot's anchor frame. |

```python
from genelab.managers import ObservationGroupCfg, ObservationTermCfg
from genelab import mdp
from genelab.mdp.noise import Unoise

observations_cfg = {
    "policy": ObservationGroupCfg(
        terms={
            "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel,
                                                noise=Unoise(n_min=-0.1, n_max=0.1)),
            "joint_pos_rel": ObservationTermCfg(func=mdp.joint_pos_rel),
            "last_action": ObservationTermCfg(func=mdp.last_action),
            "commands": ObservationTermCfg(func=mdp.generated_commands,
                                            params={"command_name": "base_velocity"}),
        },
        enable_corruption=True,
    ),
}
```

## Rewards

All reward functions return shape `(B,)`. `RewardManager` multiplies by
`RewardTermCfg.weight` and (when `scale_rewards_by_dt=True`) by `dt`. Negative
weights turn a "deviation" reward into a penalty.

| Name | Params | Behavior |
|---|---|---|
| `track_linear_velocity_xy_exp` | `command_name`, `std=0.5` | `exp(-||cmd_xy − vel_xy||² / std²)`. |
| `track_angular_velocity_z_exp` | `command_name`, `std=0.5` | `exp(-(cmd_z − vel_z)² / std²)`. |
| `action_rate_l2` | — | `Σ_d (action_d − prev_action_d)²` — penalize action jitter. |
| `joint_acc_l2` | — | **Placeholder** (currently returns zeros; awaits accel buffer). |
| `flat_orientation_l2` | — | `Σ (projected_gravity_xy)²` — penalize tilt. |
| `upright_exp` | `std=0.45` | `exp(-||projected_gravity_xy||² / std²)` — positive reward for upright base. |
| `variable_posture` | `command_name`, `std_standing` / `std_walking` / `std_running` (regex→float dicts), `default_std`, `walking_threshold`, `running_threshold` | Speed-dependent posture reward: `exp(-mean((joint_pos − default)² / std²))`, with std chosen per env from the command magnitude. Class-style term — instantiated once at construct, then `__call__` per step. |
| `joint_pos_limits` | — | L2 of joint-position excursion past ±π. |
| `feet_air_time` | `threshold=0.4` | **Stub** — reward proportional to mean foot-link height above ground, clamped at `threshold`. |
| `motion_global_anchor_position_error_exp` | `command_name`, `std` | `exp(-||p_ref − p_robot||² / std²)` on the anchor body in world frame. |
| `motion_global_anchor_orientation_error_exp` | `command_name`, `std` | Geodesic rotation error on the anchor body, mapped through a Gaussian kernel. |
| `motion_relative_body_position_error_exp` | `command_name`, `std`, `body_names=None` | Multi-body L2 position error against anchor-aligned reference frames. `body_names=None` includes every tracked body. |
| `motion_relative_body_orientation_error_exp` | `command_name`, `std`, `body_names=None` | Multi-body geodesic rotation error against anchor-aligned references. |
| `motion_global_body_linear_velocity_error_exp` | `command_name`, `std`, `body_names=None` | L2 world-frame linear-velocity error across the tracked bodies. |
| `motion_global_body_angular_velocity_error_exp` | `command_name`, `std`, `body_names=None` | L2 world-frame angular-velocity error across the tracked bodies. |

!!! warning "Placeholder rewards"
    `joint_acc_l2` and `feet_air_time` are stubs awaiting proper accel /
    contact buffers. They compile and emit a tensor of the correct shape but
    the *value* is not the locomotion-grade signal one would normally expect
    from these names. Treat them as `weight=0.0` placeholders until they are
    wired against per-step velocity history (`joint_acc_l2`) and a
    `ContactSensor` (`feet_air_time` — `mdp.foot_air_time` already provides
    the real signal as an observation term).

```python
from genelab.managers import RewardTermCfg
from genelab import mdp

rewards_cfg = {
    "track_lin_vel": RewardTermCfg(
        func=mdp.track_linear_velocity_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    ),
    "track_ang_vel": RewardTermCfg(
        func=mdp.track_angular_velocity_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.25},
    ),
    "upright":       RewardTermCfg(func=mdp.upright_exp,         weight=0.2),
    "action_rate":   RewardTermCfg(func=mdp.action_rate_l2,      weight=-0.005),
    "flat_ori":      RewardTermCfg(func=mdp.flat_orientation_l2, weight=-0.5),
}
```

## Terminations

All termination functions return shape `(B,)` bool. `TerminationTermCfg.time_out`
routes the value to the truncation buffer (RSL-RL's "info['time_out']") instead
of the terminated buffer.

| Name | Params | Behavior |
|---|---|---|
| `time_out` | — | `episode_length_buf >= max_episode_length`. Always paired with `time_out=True`. |
| `bad_orientation` | `limit_angle=math.radians(70.0)` | True when the body z-axis tilts more than `limit_angle` from world up. |
| `root_height_below` | `min_height` | True when `root_pos.z < min_height`. |
| `bad_anchor_pos_z_only` | `command_name`, `threshold` | True when the robot anchor z drifts further than `threshold` from the reference clip. |
| `bad_anchor_ori` | `command_name`, `threshold` | True when the tilt error between robot anchor and reference exceeds `threshold` (gravity-z proxy). |
| `bad_motion_body_pos_z_only` | `command_name`, `threshold`, `body_names=None` | True when any selected body's vertical position deviates past `threshold`. |

```python
from genelab.managers import TerminationTermCfg
from genelab import mdp

terminations_cfg = {
    "time_out":  TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=mdp.bad_orientation,
                                     params={"limit_angle": 1.0}),
}
```

## Events

Event functions take an extra `env_ids` argument and return `None`. The
`EventTermCfg.mode` decides when they fire — `startup` once at construct,
`reset` on every env reset, `interval` on per-env countdown (requires
`interval_range_s`).

| Name | Params | Behavior |
|---|---|---|
| `reset_root_state_uniform` | `pose_range` (`x` / `y` / `z` / `roll` / `pitch` / `yaw` → `(low, high)` dict), `velocity_range` (same axes) | Randomize floating-base pose and velocity within the given ranges. Pose offsets are added to `cfg.robot.init_pos`; orientation offsets layer on top of `cfg.robot.init_quat`. |
| `reset_joints_to_default` | `pos_jitter=0.0`, `vel_jitter=0.0` | Write the default joint pose to the selected envs, optionally with uniform ±jitter on position and velocity. |
| `push_by_setting_velocity` | `velocity_range` (`x` / `y` / `z` / `roll` / `pitch` / `yaw` → `(low, high)`) | Overwrite the base linear and angular velocity. Combined with `mode="interval"`, this is the canonical "random push" disturbance. |

```python
from genelab.managers import EventTermCfg
from genelab import mdp

events_cfg = {
    "reset_root": EventTermCfg(
        mode="reset",
        func=mdp.reset_root_state_uniform,
        params={"pose_range": {"yaw": (-3.14, 3.14)},
                "velocity_range": {"x": (-0.1, 0.1)}},
    ),
    "push_robot": EventTermCfg(
        mode="interval",
        func=mdp.push_by_setting_velocity,
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    ),
}
```

## Curriculums

| Name | Params | Behavior |
|---|---|---|
| `terrain_levels_vel` | `distance_threshold`, `demote_ratio=0.5` | Promote / demote each env's `TerrainImporter.terrain_levels` index by how far it walked from spawn. Envs that travelled more than `distance_threshold` move up; envs that travelled less than `distance_threshold * demote_ratio` move down. Levels then drive a fresh spawn-origin lookup, and the curriculum writes the new root pose into the sim. No-op when `env.scene.terrain is None`. Returns the per-env mean level so the manager logs `Curriculum/<term-name>`. |

```python
from genelab.managers import CurriculumTermCfg
from genelab.mdp.curriculums import terrain_levels_vel

curriculum_cfg = {
    "terrain_levels": CurriculumTermCfg(
        func=terrain_levels_vel,
        params={"distance_threshold": 5.0, "demote_ratio": 0.5},
    ),
}
```

## Noise

`NoiseCfg` is the abstract base; concrete subclasses plug into
`ObservationTermCfg.noise` and only fire when the enclosing
`ObservationGroupCfg.enable_corruption=True`.

| Name | Fields | Behavior |
|---|---|---|
| `NoiseCfg` | — | Abstract base; subclass and implement `apply(data) -> Tensor`. |
| `Unoise` | `n_min=-1.0`, `n_max=1.0` | Uniform additive noise sampled in `[n_min, n_max]`. |
| `Gnoise` | `mean=0.0`, `std=1.0` | Gaussian additive noise `N(mean, std²)`. |

```python
from genelab.managers import ObservationGroupCfg, ObservationTermCfg
from genelab.mdp.noise import Gnoise, Unoise
from genelab import mdp

observations_cfg = {
    "policy": ObservationGroupCfg(
        enable_corruption=True,
        terms={
            "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel,
                                                noise=Unoise(n_min=-0.1, n_max=0.1)),
            "joint_pos_rel": ObservationTermCfg(func=mdp.joint_pos_rel,
                                                  noise=Gnoise(std=0.01)),
        },
    ),
}
```

## See also

- [Managers and MDP terms](managers.md)
- [Sensors](sensors.md)
- [API Reference](../api/reference.md)
