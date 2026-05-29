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

## Robot description formats

`ArticulationCfg` accepts either MJCF or URDF as the source description:

- `mjcf_path` — absolute path to a MuJoCo `.xml`. Loaded via `gs.morphs.MJCF`.
- `urdf_path` — absolute path to a `.urdf`, `.xacro`, or `.urdf.xacro`. Loaded via
  `gs.morphs.URDF`, which preprocesses xacro files through the `xacro` Python package
  at load time. Set `xacro_args={"key": "value", ...}` to override `<xacro:arg>`
  declarations.

Exactly one of the two paths must be set; mixing both raises a `ValueError` at
`Articulation` construction. `xacro_args` is only valid with `urdf_path`.

## Why wrappers matter

Genesis APIs and Isaac Lab-style task code use different vocabulary. Wrappers isolate that
difference: MDP terms read stable GeneLab properties while the backend integration handles Genesis
details.

## Avatar (kinematic) entities

Use the `Avatar` entity when a body should appear in the scene visually but
**not** participate in physics — typical use cases are scripted motion-clip
ghosts, target reference markers, or a "follow-the-leader" demonstrator that
the env writes a pose to each step without exerting force on the rest of the
scene.

```python
from genelab.entity import Avatar, AvatarCfg

ghost = Avatar(
    AvatarCfg(morph="box", size=(0.1, 0.1, 0.1), init_pos=(0.0, 0.0, 1.0)),
    name="motion_ghost",
)
ghost.spawn(scene)
# every env step:
ghost.set_pose(target_pos, target_quat)
```

The avatar is backed by Genesis 1.0's `Kinematic` material so the rigid
solver leaves it alone (no contacts, no constraints), but sensors and cameras
still see it.

## Rasterizer toggle for batched rendering

`InteractiveSceneCfg.use_rasterizer` (default `False`) controls which
backend Genesis's `BatchRenderer` uses when `batch_render=True`:

- `False` — raytracer (default; higher fidelity, slower).
- `True` — rasterizer (substantially faster batched rendering when
  photorealism isn't required).

Ignored when `batch_render=False`, since no `BatchRenderer` is constructed.

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
