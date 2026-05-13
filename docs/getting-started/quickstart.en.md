# Quickstart

This walks through the shortest path from a fresh `uv sync` to running a registered task.

## List what's available

The core `genelab` package ships with empty registries — robots, environments, and tasks come from
extension packages. The example extension under `examples/genelab_examples/` is discovered
automatically because its `pyproject.toml` declares a `genelab.extensions` entry point.

```bash
uv run genelab list robots
uv run genelab list envs
uv run genelab list tasks
```

If a registry is empty, install or import an extension (see [Extensions](../concepts/extensions.md)).

## Play a task

```bash
uv run genelab play <task-id>
```

The CLI loads the registered factory for `<task-id>`, applies any config overrides you pass on the
command line, and runs the rollout in the configured Genesis backend.

Common shortcuts:

```bash
uv run genelab play <task-id> --vis           # enable visualization
uv run genelab play <task-id> --steps 500     # cap the episode length
uv run genelab play <task-id> --gpu 1         # pin to a single GPU
```

For arbitrary overrides, use the dotted `--a.b.c VALUE` syntax — strings are coerced to the type
declared on the target dataclass field. See [Play and Train](../cli/play-train.md) for the full
override grammar.

## Train (when a runner exists)

If a task ships with an RL runner (rsl_rl, etc.), launch training with:

```bash
uv run genelab train <task-id>
uv run genelab train <task-id> --gpus 4       # multi-GPU via torchrun
uv run genelab train <task-id> --checkpoint path/to/model.pt
```

The `--gpus N` flag transparently dispatches through `torchrun` for distributed training; the
target task's runner needs to be `torchrun`-compatible.

## Start a downstream project

When you outgrow the example extension and want your own package:

```bash
uv run genelab project new my_robot_project
```

This scaffolds `config.py`, `robots.py`, `envs.py`, `tasks.py`, and a `pyproject.toml` with a
`genelab.extensions` entry point. See [Project new](../cli/project-new.md).

## Next steps

- [CLI overview](../cli/overview.md) — all subcommands and global flags.
- [Registry](../concepts/registry.md) — how `ROBOTS` / `ENVS` / `TASKS` work.
- [Configs](../concepts/configs.md) — `ManagerBasedEnvCfg` and `apply_overrides`.
- [Extensions](../concepts/extensions.md) — three ways to register downstream code.
