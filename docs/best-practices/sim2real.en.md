# How to Harden a Policy for Sim2Real

This guide is the deployment recipe: **which domain randomization (DR) and noise to enable
while training, what to dump at export, and how to align the policy on the hardware side.**
It assumes you can already train and replay a task (see *Run RL Experiments*).

The goal is a policy that survives the reality gap — inaccurate inertia, friction, PD gains,
sensor bias, and actuator imperfections — without being trained on the real robot.

## 1. Enable domain randomization while training

DR events are wired into `events_cfg` and follow the
[`EventTermCfg`](../reference/configuration.md) calling convention. Sample **once per
episode** with `mode="startup"`; sample **mid-episode** with `mode="interval"` (M2.2).

| DR event (`genelab.mdp.dr` / `mdp.events`) | What it perturbs |
|---|---|
| `geom_friction` | per-link friction coefficient (ground / feet) |
| `body_com_offset` | centre-of-mass position (inertial calibration error) |
| `body_mass_offset` | per-link mass (kg) |
| `randomize_joint_stiffness_damping` | per-env PD gains (kp / kv) — sim-side for implicit-PD, Python gain scale for force-channel |
| `randomize_actuator_deadzone` | per-joint torque deadzone (stiction / backlash) |
| `encoder_bias` | constant joint-angle offset (silently shifted zero-point) |
| `push_by_setting_velocity` (`mdp.events`) | impulse base-velocity kick — use `mode="interval"` |

A representative startup block (mirrors `examples/unitree/.../g1/env_cfg.py`):

```python
from genelab import mdp
from genelab.managers import EventTermCfg
from genelab.managers.scene_entity_cfg import SceneEntityCfg

events_cfg = {
    "foot_friction": EventTermCfg(
        mode="startup",
        func=mdp.dr.geom_friction,
        params={"asset_cfg": SceneEntityCfg(name="robot", link_names=("left_foot", "right_foot")),
                "ranges": (0.3, 1.2), "shared_random": True},
    ),
    "base_com": EventTermCfg(
        mode="startup",
        func=mdp.dr.body_com_offset,
        params={"asset_cfg": SceneEntityCfg(name="robot", link_names=("torso_link",)),
                "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)}},
    ),
    "pd_gains": EventTermCfg(
        mode="startup",
        func=mdp.dr.randomize_joint_stiffness_damping,
        params={"stiffness_range": (0.8, 1.2), "damping_range": (0.8, 1.2)},
    ),
    "encoder_bias": EventTermCfg(
        mode="startup",
        func=mdp.dr.encoder_bias,
        params={"asset_cfg": SceneEntityCfg(name="robot"), "bias_range": (-0.015, 0.015)},
    ),
    # Mid-episode push every 5–10 s of sim time.
    "push": EventTermCfg(
        mode="interval",
        interval_range_s=(5.0, 10.0),
        func=mdp.events.push_by_setting_velocity,
        params={"velocity_range": {0: (-0.5, 0.5), 1: (-0.5, 0.5)}},
    ),
}
```

Start with mild ranges and widen until same-seed return drops by no more than ~10%; that is
the budget the *Reference Runs* acceptance uses.

## 2. Corrupt observations the way real sensors do

Set `enable_corruption=True` on the **policy** observation group and attach a `NoiseCfg` per
term. Keep the **critic** group uncorrupted — it learns from clean state.

| `genelab.mdp.noise` model | Use for |
|---|---|
| `Unoise` / `Gnoise` | baseline additive sensor noise |
| `ScaledNoise` | scale-factor / gain error (grows with the signal) |
| `CorrelatedNoise` | temporally-correlated (AR(1)) noise — slow, coloured drift |
| `BiasDrift` | slowly-drifting bias (random walk, optionally clamped) |

```python
from genelab.managers import ObservationGroupCfg, ObservationTermCfg
from genelab.mdp.noise import BiasDrift, CorrelatedNoise, Unoise

policy = ObservationGroupCfg(
    enable_corruption=True,
    terms={
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(-0.01, 0.01)),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=CorrelatedNoise(std=0.5, alpha=0.8)),
        "imu_ang_vel": ObservationTermCfg(func=mdp.sensor_data,
                                          params={"sensor_name": "imu"}, noise=BiasDrift(drift_std=0.002)),
    },
)
critic = ObservationGroupCfg(enable_corruption=False, terms=policy.terms)
```

IMU sensors carry their own per-env bias — set `bias_range_lin_acc` / `bias_range_ang_acc` on
the `IMUSensorCfg` (resampled each reset) instead of layering `BiasDrift` on top.

> `CorrelatedNoise` / `BiasDrift` are **stateful** and intentionally do **not** reset on
> episode boundaries (a real drifting sensor doesn't know about resets).

## 3. Make the policy respect real actuator limits

Add hardening terms so the policy never learns behaviour the hardware can't reproduce.

* **Terminations** (`genelab.mdp`): `joint_pos_out_of_limit`, `joint_vel_out_of_limit`,
  `contact_force_limit(sensor_name, max_force)`. Set a soft velocity limit with
  `ArticulationCfg.joint_vel_limit` so `joint_vel_out_of_limit` and the `joint_vel_limits`
  reward become active.
* **Rewards** (`genelab.mdp`): `applied_torque_l2` (penalize effort), `joint_vel_limits`
  (penalize over-speed), `alive_bonus` (offset per-step penalties), `lin_vel_z_l2` /
  `base_height_l2` (discourage bouncing).

## 4. Model the actuator gap (optional)

When you have real torque-tracking logs, use `MlpResidualActuator` — a `DCMotorActuator` base
plus a TorchScript residual on `[pos_error, joint_vel]`. Train the residual net downstream and
point `MlpResidualActuatorCfg.network_file` at the saved `.pt`. With no file it degrades to a
plain `DCMotorActuator`.

## 5. Export a dependency-free policy

```bash
uv run genelab export TASK_ID logs/.../model_best.pt --format onnx --out policy.onnx
```

The export writes `policy.onnx` (a pure `nn.Module`, no rsl_rl/skrl/sb3) **and**
`policy.onnx.metadata.json`. The exported model **bakes in the obs scale + clip**, so it takes
**raw** observations and emits actions. `metadata.json` records, per obs group:

* `dim` and the ordered `terms` (each term's `name`, dim, `scale`, `clip`);
* the action `dim` / range;
* provenance (`task`, `checkpoint`).

## 6. Align the deployment side

What the hardware controller must do — and must **not** do:

* **Assemble the obs vector in `metadata.json` term order**, concatenated, as **raw** values
  (do not pre-scale — the exported model applies `scale`/`clip` internally).
* **Reconstruct the joint target** from the action: `target = default_joint_pos +
  action_scale * action`, using the same `action_scale` and default pose the env's actuators
  used. The exported model outputs the raw action only.
* **Do not replicate the sim-only corruption on hardware** — DR (§1), observation noise (§2),
  and `encoder_bias` are *training* perturbations. The real robot already has its own friction,
  bias, and sensor noise; re-adding the sim versions doubles the gap.

## 7. Validate before shipping

* `genelab eval TASK_ID model_best.pt --episodes 100` for deterministic return / success-rate
  numbers; compare against your *Reference Runs*.
* Diff `metadata.json` obs term order against your hardware obs assembly — a mis-ordered or
  mis-scaled obs vector is the most common silent deployment failure.
* Sanity-check the exported model in a plain-`torch` process (load, feed a zero obs, confirm the
  action shape matches `metadata.json`).
