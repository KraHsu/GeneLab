# MDP Terms Reference

`genelab.mdp` is a reusable term library. It does not define a task by itself; task configs select
functions and classes from this library and wire them into managers.

## Actions and commands

| Area | Public pieces |
|---|---|
| Actions | `JointPositionActionCfg`, `JointPositionAction` |
| Velocity commands | `UniformVelocityCommandCfg`, `UniformVelocityCommand` |
| Motion commands | `MotionCommandCfg`, `MotionCommand`, `MotionLoader` |

Actions convert policy outputs to simulator control. Commands hold sampled goals that observations
and rewards can read.

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

## Where to continue

- [Managers and MDP terms](managers.md)
- [Task design](../best-practices/task-design.md)
- [API Reference](../api/reference.md)
