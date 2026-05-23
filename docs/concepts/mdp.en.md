# MDP Terms Reference

`genelab.mdp` is a reusable term library. It does not define a task by itself; task configs select
functions and classes from this library and wire them into managers.

## Actions and commands

| Area | Public pieces |
|---|---|
| Actions | `JointPositionActionCfg`, `JointPositionAction` |
| Cartesian actions | `DifferentialIKActionCfg`, `DifferentialIKAction`, `BinaryGripperActionCfg`, `BinaryGripperAction` |
| Velocity commands | `UniformVelocityCommandCfg`, `UniformVelocityCommand` |
| Motion commands | `MotionCommandCfg`, `MotionCommand`, `MotionLoader` |

Actions convert policy outputs to simulator control. Commands hold sampled goals that observations
and rewards can read.

### Action terms

| Term | Action dim | Description |
|---|---:|---|
| `JointPositionAction` | matched joints | Maps policy output to joint-position targets for the selected joints. |
| `DifferentialIKAction` | 3 or 6 | Converts end-effector deltas into arm joint targets with damped-least-squares Jacobian IK. |
| `BinaryGripperAction` | 1 | Snaps matched finger joints to an open or closed target from one scalar action. |

`DifferentialIKActionCfg.body_name` selects the controlled link, and `joint_names` selects the
arm joints used as IK columns. With `use_orientation=False`, the policy emits `(dx, dy, dz)`;
with `use_orientation=True`, it emits `(dx, dy, dz, droll, dpitch, dyaw)` as an axis-angle delta.
`scale` caps the physical delta per control step, `damping` regularizes the IK solve, and
`max_delta_joint` keeps each joint update local.

The robot articulation must set `requires_jac_and_ik=True` before a `DifferentialIKAction` term is
constructed, otherwise Genesis does not allocate the Jacobian data needed by the solver.

`BinaryGripperActionCfg` uses `threshold` to choose between `closed_pos` and `open_pos`. It is often
paired with a Cartesian arm term; for example, the Franka Cartesian pick-and-place task composes
`DifferentialIKAction(body_name="hand")` with `BinaryGripperAction` to expose a 4-D
`(dx, dy, dz, gripper)` action space.

## Observations

Common observation functions include base velocity, projected gravity, relative joint position and
velocity, last action, generated commands, sensor data, contact features, terrain height scans, and
motion-tracking state.

Observation functions should return `(num_envs, d)` or `(num_envs,)` tensors. The observation
manager handles optional noise, scaling, clipping, and group concatenation.

## Rewards and terminations

Reward functions cover velocity tracking, action smoothness, joint acceleration, orientation,
limits, foot clearance, slip, air time, self collision, angular momentum, and motion-tracking
errors. Termination functions cover time-out, orientation, root height, and motion-tracking failure.

Reward functions should return `(num_envs,)`; termination functions should return `(num_envs,)`
boolean tensors.

## Events, curricula, metrics, and noise

| Area | Examples |
|---|---|
| Events | `reset_root_state_uniform`, `reset_joints_to_default`, `push_by_setting_velocity` |
| Curricula | `terrain_levels_vel`, `commands_vel` |
| Metrics | `mean_action_acc`, `angular_momentum_mean`, `air_time_mean`, `slip_velocity_mean` |
| Noise | `Unoise`, `Gnoise` |
| Domain randomization | `mdp.dr.body`, `mdp.dr.joint`, `mdp.dr.geom` |

`NoiseCfg`'s canonical home is `genelab.contracts`, which lets the observation manager type-hint
noise without importing `genelab.mdp`. Concrete models remain available from `genelab.mdp.noise`
and the public facades.

## Where to continue

- [Managers and MDP terms](managers.md)
- [Franka Pick-and-Place](../examples/franka-pick-and-place.md)
- [Task design](../best-practices/task-design.md)
- [API Reference](../api/reference.md)
