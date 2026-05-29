# Scene and Entities

`InteractiveScene` is GeneLab's Genesis scene owner. It translates declarative configs into live
Genesis handles and exposes Isaac Lab-shaped entity wrappers to the rest of the environment.

## Scene boundary

`InteractiveSceneCfg` describes what should exist: env spacing, entities, terrain, sensors,
recordings, viewer interaction, and batch rendering. `InteractiveScene` owns what actually exists:
the Genesis `Scene`, articulations, rigid objects, sensors, terrain importer, recorder bridge, and
viewer state.

This separation keeps configs serializable and lets tasks be inspected before Genesis starts.

## Entities

| Entity | Purpose |
|---|---|
| `Articulation` | Robot wrapper with joint/link names, default joint state, limits, and refreshed `RobotState`. |
| `RigidObject` | Non-articulated object wrapper. |
| `RobotState` | Batched tensors read by observations, rewards, sensors, and events. |

`ManagerBasedRlEnv` adds the configured robot as the `"robot"` articulation. Access live robot data
through the named entity table, for example `env.articulations["robot"].data` for `RobotState` and
`env.articulations["robot"].joint_names` for joint metadata. The environment also exposes
`env.scene` for scene-level access.

## Why wrappers matter

Genesis APIs and Isaac Lab-style task code use different vocabulary. Wrappers isolate that
difference: MDP terms read stable GeneLab properties while the backend integration handles Genesis
details.

## Camera debug helpers

`InteractiveScene` exposes two thin wrappers around Genesis 1.0's debug-draw
API for inspecting attached cameras:

- `draw_camera_frustums(camera_names=None, color=(1, 1, 1, 0.3))` draws the view
  frustum of every (or named) `CameraSensor` on the scene. Returns the number of
  frustums drawn. Raises if a name does not resolve to a `CameraSensor`.
- `draw_camera_trajectory(positions, radius=0.002, color=(1, 0.5, 0, 0.8))`
  draws a polyline through `positions` (one row per world-frame point). The
  caller owns the buffer — typically a recorded camera path — so the scene
  keeps no history.

Both helpers must be called after `InteractiveScene.build()`; calling pre-build
raises `RuntimeError`. Use them from a play-time hook to visualise where the
RGB-D sensors are looking and how their pose evolved over a rollout.

## Where to continue

- [Module Map](../reference/module-map.md)
- [Sensors](sensors.md)
- [Actuators](actuators.md)
