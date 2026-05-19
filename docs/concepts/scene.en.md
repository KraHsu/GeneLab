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

`ManagerBasedRlEnv` adds the configured robot as the `"robot"` articulation and exposes convenient
properties such as `env.robot_state`, `env.joint_names`, `env.link_names`, and `env.scene`.

## Why wrappers matter

Genesis APIs and Isaac Lab-style task code use different vocabulary. Wrappers isolate that
difference: MDP terms read stable GeneLab properties while the backend integration handles Genesis
details.

## Where to continue

- [Module Map](../reference/module-map.md)
- [Sensors](sensors.md)
- [Actuators](actuators.md)
