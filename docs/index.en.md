# GeneLab

GeneLab is an Isaac Lab-inspired API for RL and robotics research powered by
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis). It keeps the familiar shape of
registered robots, environments, tasks, manager-based MDP configuration, and CLI dispatch,
while using Genesis as the simulation backend.

## Goals

- Small registries for robots, environments, and tasks.
- Core API layers separated from example assets and demo scripts.
- Manager-style config hooks for actions, observations, rewards, events, and terminations.
- Explicit, easy-to-extend Genesis backend integration.
- A stable package layout and CLI for downstream robotics projects.

## Requirements

- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/) for dependency management.

## Modules at a glance

- `genelab.registry` — registries, registration helpers, and extension loading.
- `genelab.configs` — reusable dataclass configs, including `ManagerBasedEnvCfg` and `TaskCfg`.
- `genelab.lab` — public facade for registry and manager-based environment primitives.
- `genelab.envs` / `genelab.robots` / `genelab.tasks` — core registry helper namespaces.
- `genelab.actuator` / `genelab.entity` / `genelab.scene` / `genelab.sensor` /
  `genelab.terrains` / `genelab.rl` — extension namespaces for robotics research code.

## See also

- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)
- [API Reference](api/reference.md)
