# Play and Train

`play` runs a registered task. `train` runs a registered task through a supported runner when the
task provides an agent config.

## Play

```bash
uv run genelab play TASK_ID --steps 128
uv run genelab play TASK_ID --vis --steps 500
uv run genelab play TASK_ID --agent random --steps 128
```

Policy sources:

| Agent | Behavior |
|---|---|
| `zero` | Zero actions. Default when no checkpoint is given. |
| `random` | Uniform random actions in `[-1, 1]`. |
| `trained` | Load a checkpoint and use the runner inference policy. |

Checkpoint replay:

```bash
uv run genelab play TASK_ID \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_300.pt
```

## Train

```bash
uv run genelab train TASK_ID --num_envs 4096 --max_iterations 300
```

For distributed training:

```bash
uv run genelab train TASK_ID --gpus 4 --num_envs 4096
```

`--num_envs` is total across ranks and must divide evenly by `--gpus`. Use
`--num_envs_per_gpu` for per-rank semantics.

## RL backends

The training backend is chosen automatically from the type of the task's agent
config — no flag required:

| Agent config | Backend | Algorithms |
|---|---|---|
| `RslRlOnPolicyRunnerCfg` | `rsl_rl` (default) | PPO |
| `SkrlAgentCfg` | `skrl` | PPO, A2C, SAC, TD3, DDPG |

The [skrl](https://skrl.readthedocs.io) backend is optional — install it with the
`skrl` extra (`uv sync` already includes it in this checkout; downstream users run
`pip install genelab[skrl]`). Pick the algorithm via `SkrlAgentCfg.algorithm`.

skrl trains in environment **timesteps** rather than learning iterations, so
`--max_iterations N` sets the skrl timestep budget for a skrl task. Multi-GPU
(`--gpus`) is supported by the RSL-RL backend only.

```bash
# A task registered with a SkrlAgentCfg routes through the skrl backend.
uv run genelab train GeneLab-Franka-Pick-And-Place-skrl-v0 --num_envs 2048 --max_iterations 12000
```

## Config overrides

Any unknown option after the task id is treated as a dotted config override:

```bash
uv run genelab play TASK_ID \
  --env.simulation.dt 0.005 \
  --env.rewards_cfg.action_rate.weight -0.01
```

## See also

- [Run RL Experiments](../best-practices/rl-experiments.md)
- [CLI Reference](../reference/cli.md)
- [RL runner](../concepts/rl-runner.md)
