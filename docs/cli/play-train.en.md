# Play and Train

`play` and `train` share a common dispatch path: resolve `<task-id>` against the `TASKS` registry,
construct its `TaskCfg`, apply any command-line overrides, and hand the configured task to either
a single rollout (`play`) or the task's RL runner (`train`).

## Play

```bash
uv run genelab play <task-id> [SHORT FLAGS] [-- OVERRIDES]
```

### Short flags

These three short flags rewrite to the corresponding `env.scene.*` overrides:

| Flag | Equivalent override | Effect |
|------|--------------------|--------|
| `--vis` | `env.scene.vis=true` | Enable Genesis visualization. |
| `--gpu N` | `env.scene.gpu=N` | Pin the rollout to a single GPU index. |
| `--steps N` | `env.scene.steps=N` | Cap episode length to `N` steps. |

### Override grammar

After the short flags, any `--<dotted.path> VALUE` argument is parsed as a config override:

```bash
uv run genelab play <task-id> \
  --env.scene.dt 0.005 \
  --env.actions.scale 0.5 \
  --env.observations.include_velocity true
```

The dotted path walks the `TaskCfg` dataclass tree (typically rooted at `env.*`). Values are
coerced from string using the field's type hint — `int`, `float`, `bool`, `Path`, `list[...]`,
`tuple[...]` are all supported. See [Configs](../concepts/configs.md) for details on the coercion
rules and how to handle unions and `Literal`.

!!! tip "Flag ordering"
    The CLI's `_normalize_run_flags` rewrites `play --steps 5 <task-id> ...` into
    `play <task-id> --steps 5 ...` so that `argparse.REMAINDER` works. You can keep the short
    flags before or after the task ID.

## Train

```bash
uv run genelab train <task-id> [SHORT FLAGS] [-- OVERRIDES]
```

Train requires the registered task to expose an RL runner (commonly `rsl_rl_lib`). Override syntax
is identical to `play`. Additional flags:

| Flag | Effect |
|------|--------|
| `--gpus N` | Dispatch via `torchrun` for `N`-process distributed training. |
| `--checkpoint PATH` | Resume training from a checkpoint file. |

### Multi-GPU

`--gpus N` wraps the underlying training entry point with `torchrun --standalone --nproc_per_node=N`.
The task's runner must be `torchrun`-compatible; environments that depend on global Genesis state
typically are.

When `--gpus N` is set, the CLI also masks `CUDA_VISIBLE_DEVICES` to the requested devices so that
each rank sees a distinct GPU.

## Examples

```bash
# Quick local rollout with visualization.
uv run genelab play wuji_hand --vis --steps 200

# Train on 4 GPUs with a custom action scale.
uv run genelab train wuji_hand --gpus 4 --env.actions.scale 0.3

# Resume from a checkpoint.
uv run genelab train wuji_hand --checkpoint logs/wuji_hand/run_42/model_100.pt
```

## See also

- [Configs](../concepts/configs.md) — full `apply_overrides` semantics.
- [Registry](../concepts/registry.md) — how tasks resolve from `<task-id>`.
