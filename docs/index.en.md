# GeneLab

GeneLab is an Isaac Lab-inspired API for RL and robotics research powered by
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis). It keeps the familiar shape of registered
robots, environments, tasks, manager-based MDP configuration, and CLI dispatch, while using Genesis
as the simulation backend.

## Goals

- Provide small registries for robots, environments, and tasks.
- Keep core API layers separate from example assets and demo scripts.
- Use manager-style config hooks for actions, observations, rewards, events, and terminations.
- Keep Genesis backend integration explicit and easy to extend.
- Support downstream robotics projects through a stable package layout and CLI.

## Quick start

<div class="grid cards" markdown>

- :material-download:{ .lg .middle } **Install**

    ---

    Set up `uv`, pick a `torch-*` extra, and verify.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

- :material-rocket-launch-outline:{ .lg .middle } **Run**

    ---

    List registered tasks and play one.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

- :material-console-line:{ .lg .middle } **CLI**

    ---

    `play`, `train`, `project new`, and override syntax.

    [:octicons-arrow-right-24: CLI overview](cli/overview.md)

- :material-package-variant-closed:{ .lg .middle } **Extend**

    ---

    Write a downstream extension package.

    [:octicons-arrow-right-24: Extensions](concepts/extensions.md)

</div>

## Requirements

- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/) for dependency management.

## At a glance

- `genelab.registry` — registries, registration helpers, and extension loading.
- `genelab.configs` — reusable dataclass configs, including `ManagerBasedEnvCfg` and `TaskCfg`.
- `genelab.lab` — public facade for registry and manager-based environment primitives.
- `genelab.envs` / `genelab.robots` / `genelab.tasks` — core registry helper namespaces.
- `genelab.actuator` / `genelab.entity` / `genelab.scene` / `genelab.sensor` / `genelab.terrains` /
  `genelab.rl` — extension namespaces for robotics research code.

See the [API Reference](api/reference.md) for the full auto-generated module documentation.
