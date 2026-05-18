# Discovery: list and info

Use `list` to see what extensions registered. Use `info` to inspect one registered object and copy
override paths.

## List registries

```bash
uv run genelab list robots
uv run genelab list envs
uv run genelab list tasks
```

If an extension is not installed, import it explicitly:

```bash
PYTHONPATH=examples/inverted_pendulum/src \
  uv run genelab --import genelab_inverted_pendulum.tasks list tasks
```

## Inspect a task

```bash
uv run genelab info GeneLab-Inverted-Pendulum-v0
```

The task view includes:

| Section | Meaning |
|---|---|
| Metadata | Registered name, description, cfg type, examples. |
| Task config | `TaskCfg` summary: env, robot, trainable flag, agent. |
| Overridable paths | Dotted paths accepted by `play` and `train`. |

## Use copied override paths

```bash
uv run genelab play GeneLab-Inverted-Pendulum-v0 \
  --env.rewards_cfg.pole_upright.weight 4.0
```

When `play_env` exists, play-mode shortcut flags target `play_env`; explicit paths remain explicit.

## See also

- [Configuration Reference](../reference/configuration.md)
- [CLI Reference](../reference/cli.md)
