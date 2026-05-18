# Managers and MDP Terms

GeneLab follows the Isaac Lab-style manager-based environment pattern: the environment owns the
simulation loop, while MDP behavior is split into named terms managed by specialized managers.

## Why split the MDP

A robot learning environment usually contains many concerns:

- action decoding
- command sampling
- observations
- rewards
- terminations
- reset and interval events
- curricula
- metrics

Putting all of that directly in `step()` makes tasks hard to inspect and override. Manager-based
configs keep each concern named, typed, and discoverable.

## The manager set

| Manager | Config field | Role |
|---|---|---|
| `ActionManager` | `actions_cfg` | Converts policy actions into simulator targets or torques. |
| `CommandManager` | `commands_cfg` | Maintains sampled goals such as target velocity or reference motion. |
| `ObservationManager` | `observations_cfg` | Computes named observation groups such as `policy` and `critic`. |
| `RewardManager` | `rewards_cfg` | Computes weighted reward terms and episode summaries. |
| `TerminationManager` | `terminations_cfg` | Separates terminated and time-out conditions. |
| `EventManager` | `events_cfg` | Runs startup, reset, and interval randomization or perturbation. |
| `CurriculumManager` | `curriculum_cfg` | Adjusts difficulty or task state at reset boundaries. |
| `MetricsManager` | `metrics_cfg` | Records non-reward diagnostics. |

## Runtime order

`ManagerBasedRlEnv` builds the Genesis scene first, then creates managers, binds sensors, applies
startup events, and runs an initial reset. During each env step it processes actions, advances the
scene for `decimation` physics ticks, refreshes state, updates sensors, computes commands, events,
rewards, metrics, terminations, resets done envs, and finally computes observations.

This order matters because terms read shared state. For example, reward terms see the state after
actions and sensor updates; reset events run before the next observation is emitted.

## Term names are part of the interface

Term names appear in override paths and logs. A reward named `track_lin_vel` can be changed from the
CLI and appears in episode summaries. Stable names make experiments easier to compare.

## Where to continue

- [Design a Task](../best-practices/task-design.md)
- [MDP terms reference](mdp.md)
- [API Reference](../api/reference.md)
