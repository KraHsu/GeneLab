# Managers and MDP terms

`genelab.managers` exposes the seven manager-style MDP hooks that
`ManagerBasedRlEnv` orchestrates: actions, commands, observations, rewards,
terminations, events, and curriculums. Each manager owns a dict of *term* cfgs
keyed by name; the env constructs them in a fixed order, then drives them once
per control step or once per reset.

## Term and manager

A *term* is a single MDP-side computation — one reward component, one
observation column group, one termination check. A *manager* is the container
that instantiates all terms of one kind, exposes their aggregate output, and
schedules their per-step / per-reset calls.

```
ManagerBasedRlEnvCfg
├── actions_cfg:       dict[str, ActionTermCfg]
├── commands_cfg:      dict[str, CommandTermCfg]
├── observations_cfg:  dict[str, ObservationGroupCfg]   # group → terms
├── rewards_cfg:       dict[str, RewardTermCfg]
├── terminations_cfg:  dict[str, TerminationTermCfg]
├── events_cfg:        dict[str, EventTermCfg]
└── curriculum_cfg:    dict[str, CurriculumTermCfg]
```

Every term cfg inherits `ManagerTermBaseCfg` with two fields:

| Field | Type | Meaning |
|---|---|---|
| `func` | `Callable[..., Any]` | The per-step computation. For observation / reward / termination / event / curriculum terms this is a function taking `(env, **params)` and returning a tensor (or `None` for events). Action and command terms ignore `func` and use `class_type` instead. |
| `params` | `dict[str, Any]` | Keyword arguments injected on every call. `apply_overrides` can mutate individual entries via dotted paths. |

## The seven managers

| Manager | Term cfg | Term identifier | Call schedule |
|---|---|---|---|
| `ActionManager` | `ActionTermCfg(class_type=…)` | `class_type` | `process_action` once per env step; `apply_action` `decimation` times per env step |
| `CommandManager` | `CommandTermCfg(class_type=…)` | `class_type` | `compute(dt)` once per env step (resampling on countdown) |
| `ObservationManager` | `ObservationGroupCfg.terms[name] = ObservationTermCfg(func=…)` | `func` | `compute()` once per env step (after reset and after step) |
| `RewardManager` | `RewardTermCfg(func=…, weight=…)` | `func` | `compute(dt)` once per env step |
| `TerminationManager` | `TerminationTermCfg(func=…, time_out=…)` | `func` | `compute()` once per env step |
| `EventManager` | `EventTermCfg(func=…, mode=…)` | `func` | once per construct (`startup`), once per reset (`reset`), or on per-env countdown (`interval`) |
| `CurriculumManager` | `CurriculumTermCfg(func=…)` | `func` | `compute(env_ids)` once per reset, after the other managers reset |

## Term lifecycle

`ManagerBasedRlEnv.__init__` constructs the seven managers in the order above.
For each term:

1. The manager deep-copies the cfg dict so per-instance mutation never bleeds
   back into the user's source cfg.
2. If `func` is a class, `_base.instantiate_class_term` replaces it with
   `func(cfg=term_cfg, env=env)` so the term can cache references. Plain
   callables are kept as-is.
3. Action and command managers instead invoke `class_type(term_cfg, env)` and
   keep the resulting instance under the term name.

After all managers are constructed the env calls `event_manager.apply("startup")`
once, then `articulation.refresh()`, then `reset()` for every env.

## ActionManager

Each `ActionTermCfg` declares a `class_type` (an `ActionTerm` subclass) and an
optional `asset_name`. The manager concatenates per-term `action_dim` slots
into a single flat action vector. Multiple action terms can coexist, e.g. one
covering the arm joints and one covering the gripper:

```python
from genelab.mdp.actions.joint_position import JointPositionActionCfg

actions_cfg = {
    "panda_arm": JointPositionActionCfg(
        asset_name="robot",
        joint_names=(r"^joint[1-7]$",),
        use_default_offset=True,
    ),
    "panda_hand": JointPositionActionCfg(
        asset_name="robot",
        joint_names=(r"finger_joint.*",),
        use_default_offset=True,
    ),
}
```

`process_action(action)` runs once per env step; `apply_action()` is called
`decimation` times per env step before each sim sub-step, so the controller
sees a fixed setpoint for the whole decimation window.

## ObservationManager

Observations are organised into named *groups* (`policy`, `critic`, …); each
group is a dict of named terms.

```python
from genelab.managers import ObservationGroupCfg, ObservationTermCfg
from genelab import mdp

observations_cfg = {
    "policy": ObservationGroupCfg(
        terms={
            "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
            "joint_pos_rel": ObservationTermCfg(func=mdp.joint_pos_rel),
            "last_action": ObservationTermCfg(func=mdp.last_action),
        },
        concatenate_terms=True,
        enable_corruption=False,
    ),
}
```

For each term `compute()` calls `term_cfg.func(env, **term_cfg.params)`,
unsqueezes 1-D returns to `(num_envs, 1)`, then optionally applies `noise`
(only when `enable_corruption=True`), `scale`, and `clip`. When
`concatenate_terms=True` (default) the per-group output is a single
`(num_envs, total_dim)` tensor; otherwise the manager stacks along a new last
axis.

## RewardManager

`RewardTermCfg.weight` is the signed scalar multiplier. The manager skips
zero-weight terms. When `ManagerBasedRlEnvCfg.scale_rewards_by_dt=True`
(default) the per-term tensor is also multiplied by the env step `dt`, so total
episode return is comparable across simulation frequencies.

```python
from genelab.managers import RewardTermCfg
from genelab import mdp

rewards_cfg = {
    "track_lin_vel": RewardTermCfg(
        func=mdp.track_linear_velocity_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
    "joint_acc":   RewardTermCfg(func=mdp.joint_acc_l2,   weight=-2.5e-7),
}
```

Per-term `weight * func(...)` outputs are NaN-coerced, summed, and exposed as a
single `(num_envs,)` reward tensor. The manager also accumulates a per-term
episode sum; `reset()` returns the mean episode reward per term (logged as
`Episode_Reward/<name>` in `extras["log"]`).

## TerminationManager

Each `TerminationTermCfg.func` must return a bool tensor of shape
`(num_envs,)`. Terms with `time_out=True` flow into the truncation buffer
(`info["time_out"]`); the rest accumulate into the terminated buffer. Both are
OR'd to produce the final `dones` mask. RSL-RL distinguishes the two so the
PPO update treats truncation correctly.

```python
from genelab.managers import TerminationTermCfg
from genelab import mdp

terminations_cfg = {
    "time_out":        TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over":       TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0},
    ),
}
```

## EventManager

`EventTermCfg.mode` is a `Literal["startup", "reset", "interval"]`:

| Mode | Fires |
|---|---|
| `startup` | Once, right after all managers are constructed. Useful for one-shot randomisation of physical parameters (mass, friction). |
| `reset` | On every env reset, before `command_manager.reset` and `action_manager.reset`. Use this to randomise initial joint state, root pose, or env-specific buffers. |
| `interval` | Per-env countdown sampled uniformly from `interval_range_s`. Fires inside the env step when the countdown hits zero, then resamples. Use for periodic disturbances (random push). |

```python
from genelab.managers import EventTermCfg
from genelab import mdp

events_cfg = {
    "reset_joints": EventTermCfg(
        mode="reset",
        func=mdp.reset_joints_to_default,
        params={"pos_jitter": 0.05, "vel_jitter": 0.0},
    ),
    "push_robot": EventTermCfg(
        mode="interval",
        func=mdp.push_by_setting_velocity,
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    ),
}
```

## CommandManager

A *command* is a persistent per-env target (velocity command, motion
reference, goal pose) that survives across env steps until it is resampled.
Each `CommandTermCfg.class_type` is a `CommandTerm` subclass; the term holds
the live tensor and resamples it every `resampling_time_range` seconds.
Observation and reward terms read commands back via
`env.command_manager.get_command(name)`.

```python
from genelab.mdp.commands.uniform_velocity import UniformVelocityCommandCfg

commands_cfg = {
    "base_velocity": UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0),
        ),
    ),
}
```

## CurriculumManager

Curriculum terms run on every reset (after the other managers have reset)
and may mutate scene state — `TerrainImporter.terrain_levels`, env-specific
spawn poses, reward weights. The return value is logged as
`Curriculum/<name>` in `extras["log"]`; tensor returns are reduced to their
mean across `env_ids`.

```python
from genelab.managers import CurriculumTermCfg
from genelab.mdp.curriculums import terrain_levels_vel

curriculum_cfg = {
    "terrain_levels": CurriculumTermCfg(
        func=terrain_levels_vel,
        params={
            "command_name": "base_velocity",
            "distance_threshold": 5.0,
            "demote_ratio": 0.5,
        },
    ),
}
```

## Failure modes worth knowing

!!! warning "Wrong observation tensor shape"
    `ObservationManager.compute` only auto-unsqueezes 1-D returns to
    `(num_envs, 1)`. A scalar-per-env reward term that accidentally returns a
    `()` scalar will broadcast in the wrong direction. Always return shape
    `(num_envs,)` or `(num_envs, d)`.

!!! warning "Missing class_type / func"
    `ActionManager` and `CommandManager` silently skip terms whose `class_type`
    is `None`. `RewardManager` / `TerminationManager` / `EventManager` /
    `CurriculumManager` keep the no-op default `func` from `ManagerTermBaseCfg`,
    so a missing `func=` produces a manager that returns `None` and crashes
    downstream when the result is used as a tensor. Always set the term
    factory explicitly.

!!! warning "Interval events without `interval_range_s`"
    `EventManager.__init__` asserts that every `mode="interval"` term provides
    `interval_range_s=(low, high)`. The assertion fires at manager construction
    so misconfigured tasks never reach the first reset.

!!! tip "Override term params at the CLI"
    Every term `params` entry is reachable via `apply_overrides`. The dotted
    path mirrors the cfg tree printed by `genelab info <task-id>`:
    `--env.rewards_cfg.track_lin_vel.weight 2.0` rescales a reward term;
    `--env.rewards_cfg.track_lin_vel.params.std 0.5` retargets a kwarg.

## See also

- [Configs](configs.md)
- [Sensors](sensors.md)
- [Discovery: list and info](../cli/list-info.md)
