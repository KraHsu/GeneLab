# Play and Train

`play` and `train` share a common dispatch path: resolve `<task-id>` against the `TASKS`
registry, construct its `TaskCfg`, apply any command-line overrides, and hand the configured
task to either a single rollout (`play`) or the task's RL runner (`train`).

## Play

```bash
uv run genelab play <task-id> [SHORT FLAGS] [-- OVERRIDES]
```

Short flags may appear before or after `<task-id>`; the CLI normalises the order internally.

!!! tip "Interactive recovery"

    Omitting `<task-id>` on a TTY opens a `questionary` picker over the
    registered tasks. Unknown task ids, malformed `--agent` values, and
    `--<a.b.c>` override paths that do not exist on the resolved cfg fall
    back to the same picker. Non-TTY callers — CI, pipes, scripts — see the
    original error messages unchanged.

### Simulation shortcuts

The following short flags rewrite to the corresponding `env.simulation.*` overrides:

| Flag | Equivalent override | Effect |
|------|--------------------|--------|
| `--vis` | `env.simulation.vis=true` | Enable Genesis visualization. |
| `--gpu N` | `env.simulation.gpu=N` | Pin the rollout to a single GPU index. |
| `--steps N` | `env.simulation.steps=N` | Cap episode length to `N` steps. |

### Runner flags

When the task carries an RL agent config, `play` also accepts the runner-side flags:

| Flag | Effect |
|------|--------|
| `--checkpoint PATH` | Load a trained policy and run inference rollouts. Forces `--agent` to default to `trained`. |
| `--num-envs N` | Override the registered env count (parallel environments). |
| `--agent {zero,random,trained}` | Pick the policy source. Defaults to `trained` when `--checkpoint` is set, otherwise `zero`. |

### Override grammar

After the short flags, any `--<dotted.path> VALUE` argument is parsed as a config override:

```bash
uv run genelab play <task-id> \
  --env.simulation.dt 0.005 \
  --env.actions.scale 0.5 \
  --env.observations.include_velocity true
```

The dotted path walks the `TaskCfg` dataclass tree (typically rooted at `env.*`). Values are
coerced from string using the field's type hint — `int`, `float`, `bool`, `Path`, `list[...]`,
`tuple[...]` are all supported.

## Train

```bash
uv run genelab train <task-id> [SHORT FLAGS] [-- OVERRIDES]
```

Train requires the registered task to expose an RL runner (commonly `rsl_rl_lib`). The
override syntax is identical to `play`.

| Flag | Effect |
|------|--------|
| `--gpus N` | Dispatch via `torchrun --standalone --nproc_per_node=N` for distributed training. The task's runner must be `torchrun`-compatible. |
| `--checkpoint PATH` | Resume training from a checkpoint file. |
| `--num-envs N` | Override the registered env count (parallel environments). |
| `--max-iterations N` | Cap the number of PPO learning iterations. |
| `--seed N` | Override the RNG seed used by the runner. |
| `--log-dir PATH` | Override the log root; defaults to `logs/<runner>/<experiment>/<run>`. |
| `--agent {zero,random,trained}` | Forwarded to the runner; mostly useful for diagnostic runs. |

When `--gpus N > 1`, the CLI masks `CUDA_VISIBLE_DEVICES` to the requested devices so each rank
sees a distinct GPU.

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

- [Configs](../concepts/configs.md)
- [Registry](../concepts/registry.md)
