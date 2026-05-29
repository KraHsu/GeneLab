# Discovery: list and info

Use `list` to see what extensions registered. Use `info` to inspect one registered object and copy
override paths.

## Listing registries

```bash
genelab list robots
genelab list envs
genelab list tasks
```

If an extension is not installed as an entry point, import it explicitly:

```bash
PYTHONPATH=examples/inverted_pendulum/src \
  genelab --import genelab_inverted_pendulum.tasks list tasks
```

To see only the bundled asset zoo (skip user-installed entry-point extensions):

```bash
genelab --no-entry-points list robots
```

## Inspecting any registered name

```bash
genelab info GeneLab-Inverted-Pendulum-v0   # task
genelab info CartPole-Env-v0                # env
genelab info anymal_c                       # robot
```

`info` does not require knowing which registry the name belongs to. Task views additionally
include:

| Section | Meaning |
|---|---|
| Metadata | Registered name, description, cfg type, examples. |
| Task config | `TaskCfg` summary: env, robot, trainable flag, agent. |
| Override paths | Dotted paths accepted by `play` and `train`. |

If the inspected task needs an asset that isn't cached yet, `info` fetches it inline behind the
same progress bar used by `play` / `train` / `asset download`.

## Using copied override paths

```bash
genelab play GeneLab-Inverted-Pendulum-v0 \
  --env.rewards_cfg.pole_upright.weight 4.0
```

When `play_env` exists, play-mode shortcut flags (`--vis` / `--gpu` / `--dt` / `--steps`) target
`play_env`; explicit dotted paths still write the exact subtree as written.

## See also

- [Configuration Reference](../reference/configuration.md)
- [CLI Reference](../reference/cli.md)
- [Asset zoo](../concepts/asset_zoo.md)
