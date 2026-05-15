# Scene and entities

`genelab.scene.InteractiveScene` is the composition root that owns a Genesis
`gs.Scene` together with the articulated robot, rigid props, optional terrain, and
the mouse-interaction plugin. `ManagerBasedRlEnv` defers all sim-level orchestration
to the scene; the env layer only manages the seven MDP terms.

## Composing a Genesis scene

A scene is built from two cfg objects passed to the env. `SimulationCfg` carries the
Genesis runtime knobs (timestep, parallel-env count, viewer toggle); `InteractiveSceneCfg`
carries the composition (entities, terrain, sensors, mouse plugin, BatchRenderer
toggle). The scene parses the cfg, allocates wrapper objects, then defers the actual
Genesis allocation to the `build()` call.

```python
from genelab.asset_zoo import CartpoleCfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg

env_cfg = ManagerBasedRlEnvCfg(
    simulation=SimulationCfg(num_envs=4096, dt=0.005, substeps=1, vis=False),
    scene=InteractiveSceneCfg(env_spacing=(2.5, 2.5)),
    robot=CartpoleCfg(),
)
```

## Lifecycle

`InteractiveScene` follows a strict three-step lifecycle, enforced at runtime:

1. **Construct**: `InteractiveScene(sim_cfg, scene_cfg, device_hint=...)` parses the
   cfgs and allocates wrapper objects for every entity declared in
   `scene_cfg.entities`. No Genesis import yet.
2. **Add entities**: `add_entity(name, cfg)` registers extra entities — the env uses
   this to inject `ManagerBasedRlEnvCfg.robot` under the name `"robot"`. Calling
   `add_entity` after `build()` raises `RuntimeError`.
3. **Build**: `build()` imports Genesis, initialises the backend, creates the
   `gs.Scene`, spawns ground / terrain / entities / cameras, attaches the mouse
   plugin when configured, then calls `gs_scene.build(n_envs, env_spacing)`.

`ManagerBasedRlEnv.__init__` runs all three steps in order and post-build binds every
articulation so the per-joint and per-link tensors land.

## SimulationCfg

| Field | Default | Meaning |
|---|---|---|
| `vis` | `False` | Open the Genesis viewer. Forces `num_envs=1` in practice; the viewer can only render env 0. |
| `gpu` | `False` | Use the CUDA backend (`gs.gpu`). Required for `BatchRenderer`. |
| `steps` | `240` | Default number of steps the CLI `play` runner takes when no override is given. |
| `dt` | `0.01` | Physics timestep in seconds. |
| `substeps` | `4` | Genesis substeps per physics step. |
| `num_envs` | `1` | Parallel-env count. Most managers tensorise across this dimension. |

## InteractiveSceneCfg

| Field | Default | Meaning |
|---|---|---|
| `env_spacing` | `(2.0, 2.0)` | Per-env XY offset Genesis applies when `num_envs > 1`. |
| `sensors` | `()` | Tuple of `SensorCfg` instances built and bound after the scene is constructed. |
| `mouse_interaction` | `False` | Attach the GeneLab mouse-drag plugin to the viewer (only effective with `vis=True`). |
| `entities` | `{}` | Additional `ArticulationCfg` / `RigidObjectCfg` keyed by name. The env injects the robot as `"robot"`. |
| `terrain` | `None` | A `TerrainGeneratorCfg`. Default `None` adds a flat `gs.morphs.Plane`. |
| `batch_render` | `False` | Pass `gs.renderers.BatchRenderer(use_rasterizer=False)` to `gs.Scene`. Required for `CameraSensor` to emit per-env RGB-D tensors. Linux x86-64 + CUDA only. |

## Articulation

`Articulation` wraps an articulated robot. Construct from `ArticulationCfg` and add
to the scene via `scene.add_entity("robot", cfg)` (or set
`ManagerBasedRlEnvCfg.robot` and let the env wire it).

| Field | Type | Meaning |
|---|---|---|
| `mjcf_path` | `str` | Absolute path to a MuJoCo XML file. |
| `init_pos` | `tuple[float, float, float]` | World-frame spawn offset in metres. |
| `init_quat` | `tuple[float, float, float, float]` | World-frame spawn orientation (wxyz). |
| `default_joint_pos` | `dict[str, float]` | Regex-keyed default joint position; last match wins. |
| `actuators` | `dict[str, ActuatorBaseCfg]` | Joint groups. Every actuated joint must be covered by exactly one group. |
| `foot_link_names` | `tuple[str, ...]` | Optional metadata for downstream MDP terms (e.g. contact sensors). |

The pre-M3 `joint_kp`, `joint_kv`, and dict-shaped `action_scale` knobs are
removed — configure equivalent groups through `actuators`. Passive joints take an
explicit zero-gain `ImplicitPDActuatorCfg` so the topology stays visible.

```python
from genelab.actuator import ImplicitPDActuatorCfg
from genelab.entity import ArticulationCfg

cartpole_cfg = ArticulationCfg(
    mjcf_path="/path/to/cartpole.xml",
    init_pos=(0.0, 0.0, 1.5),
    default_joint_pos={"cart_slide": 0.0, "pole_hinge": 0.0},
    actuators={
        "cart": ImplicitPDActuatorCfg(
            target_names_expr=("cart_slide",), stiffness=80.0, damping=8.0,
        ),
        "pole": ImplicitPDActuatorCfg(
            target_names_expr=("pole_hinge",), stiffness=0.0, damping=0.0,
        ),
    },
)
```

## RigidObject

`RigidObject` represents a non-articulated body. No actuators, no per-step state — it
is geometry that participates in collisions and contacts but is never driven. Spawn
via `scene.add_entity("name", RigidObjectCfg(...))` before `build()`.

| Field | Type | Meaning |
|---|---|---|
| `morph` | `Literal["plane", "box", "sphere", "mesh", "mjcf"]` | Genesis morph selector. |
| `file` | `str \| None` | Path to mesh / MJCF; required for `"mesh"` / `"mjcf"`. |
| `size` | `tuple[float, ...]` | `(x, y, z)` for box, `(radius,)` for sphere. |
| `init_pos`, `init_quat` | tuple | World-frame spawn pose (wxyz quat). |
| `fixed` | `bool` | When `True`, the body is welded to the world frame. |

Typical uses are static obstacles, target markers, and passive payloads. For
articulated joints — even one — use `Articulation` instead.

## Failure modes worth knowing

!!! warning "`add_entity` after `build`"
    `InteractiveScene.add_entity` raises `RuntimeError` once `build()` has run.
    Construct the env, then add every extra entity, then let the env call `build()`.

!!! warning "Empty `actuators` dict"
    `Articulation.bind` rejects an empty `actuators` dict. Every actuated joint must
    be covered by exactly one group; unmatched or doubly-matched joints raise
    `ValueError` with the conflicting joint names listed.

!!! warning "Genesis backend mismatch"
    `SimulationCfg.gpu=True` requires a working CUDA install. On macOS or CPU-only
    Linux, leave `gpu=False`; the CPU backend handles single-env play loops fine.
    `batch_render=True` further requires Linux x86-64 + CUDA + Madrona — there is no
    CPU fallback.

!!! note "Viewer and parallel envs"
    `SimulationCfg.vis=True` is only meaningful with `num_envs=1`. The viewer
    renders env 0 regardless, so leave `num_envs` high for training and switch to a
    dedicated `play_env` with `num_envs=1` for visual inspection.

!!! tip "Viewer closed mid-rollout"
    When the user closes the Genesis viewer, the kernel catches
    `GenesisException("Viewer closed.")` inside `InteractiveScene.step` and flips
    `env.viewer_closed = True`. Subsequent `env.step` calls become no-ops. Loop
    drivers should poll the flag instead of writing their own try / except:
    ```python
    for step in range(max_steps):
        obs, *_ = env.step(action)
        if env.viewer_closed:
            break
    ```

## See also

- [Configs](configs.md)
- [Actuators](actuators.md)
- [Asset zoo](asset_zoo.md)
