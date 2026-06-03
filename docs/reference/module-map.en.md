# Module Map

GeneLab's public modules map to the part of the system they own — useful when the change is known
but the right import path is not.

## Public facade

| Module | Use it for | Typical imports |
|---|---|---|
| `genelab.lab` | Stable, snapshot-guarded user-facing facade for configs, registries, scene objects, sensors, actuators, terrains, and extension loading. Public names are resolved lazily on first access. | `TaskCfg`, `ManagerBasedEnvCfg`, `ArticulationCfg`, `SensorCfg`, `register_*` via registries |
| `genelab` | Lightweight package root. It lazily re-exports a few top-level types without importing torch. | `__version__`, `TaskCfg`, `ManagerBasedEnvCfg` |

Prefer `genelab.lab` in notebooks, downstream tasks, and examples unless a lower-level module is
documented as the owner of the API in question.

## Discovery and dispatch

| Module | Responsibility |
|---|---|
| `genelab.registry` | `ROBOTS`, `ENVS`, `TASKS`, `RegistryEntry`, registration helpers, entry-point loading, explicit extension imports. |
| `genelab.cli` | Typer/Rich CLI: `cache`, `list`, `info`, `play`, `train`, `prof`, `project new`. |
| `genelab.cache` | Creates project-local cache directories for Genesis, Quadrants, and Matplotlib. |

## Configuration and MDP

| Module | Responsibility |
|---|---|
| `genelab.configs` | Core dataclasses: `SimulationCfg`, `InteractiveSceneCfg`, `ManagerBasedEnvCfg`, `TaskCfg`, and `apply_overrides`. |
| `genelab.contracts` | Domain ports for concrete env/scene implementations: `EnvContext`, `SceneContext`, and the canonical `NoiseCfg` base. |
| `genelab.envs.manager_based_rl_env` | Genesis-backed `ManagerBasedRlEnv` and `ManagerBasedRlEnvCfg`. |
| `genelab.managers` | Manager and term cfg types for actions, commands, observations, rewards, terminations, events, curricula, and metrics. |
| `genelab.mdp` | Reusable MDP functions for observations, rewards, terminations, events, curricula, metrics, noise, actions, commands, and domain randomization. |

## Simulation objects

| Module | Responsibility |
|---|---|
| `genelab.scene` | `InteractiveScene`, the owner of the live Genesis scene. |
| `genelab.entity` | `Articulation`, `RigidObject`, and their config dataclasses. |
| `genelab.actuator` | Implicit PD, ideal PD, and DC motor actuator models. |
| `genelab.sensor` | Base sensor API and built-in body velocity, IMU, camera, contact, ray-cast, frame transformer, terrain height, self-contact, and angular momentum sensors. |
| `genelab.terrains` | Sub-terrain configs, terrain grid generation, and terrain import into Genesis. |
| `genelab.asset_zoo` | Curated robot asset configs and motion asset fetchers. This is bundled content, not the core abstraction boundary. |

## Training and experiment I/O

| Module | Responsibility |
|---|---|
| `genelab.rl` | RSL-RL configs, VecEnv wrapper, `train_task`, `play_task`, distributed helpers, and profiler integration. |
| `genelab.recording` | Declarative recording configs for plots, files, and camera video. |
| `genelab.bridges` | Runtime input/control bridges, including keyboard twist and in-viewport ImGui twist controls. |
| `genelab.utils` | Math and asset download helpers used by higher-level modules. |

## Choosing the right layer

| Goal | Start here |
|---|---|
| Register a new robot, env, or task | `genelab.registry`, then [Build an Extension Project](../best-practices/extension-projects.md) |
| Add observations, rewards, or terminations | `genelab.managers`, `genelab.mdp`, then [Managers and MDP terms](../concepts/managers.md) |
| Add a sensor | `genelab.sensor`, then [Sensors](../concepts/sensors.md) |
| Change training or replay | `genelab.rl`, then [Run RL Experiments](../best-practices/rl-experiments.md) |
| Look up signatures and defaults | [API Reference](../api/reference.md) |
