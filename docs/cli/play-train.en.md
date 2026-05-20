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
| `Sb3AgentCfg` | `sb3` | PPO, A2C, SAC, TD3, DDPG (+ HER) |

The [skrl](https://skrl.readthedocs.io) and
[Stable-Baselines3](https://stable-baselines3.readthedocs.io) backends are
optional — install them with the `skrl` / `sb3` extras (`uv sync` already includes
both in this checkout; downstream users run `pip install genelab[skrl]` or
`genelab[sb3]`). Pick the algorithm via `SkrlAgentCfg.algorithm` /
`Sb3AgentCfg.algorithm`.

Both skrl and SB3 train in environment **timesteps** rather than learning
iterations, so `--max_iterations N` sets the timestep budget for those tasks.
Multi-GPU (`--gpus`) is supported by the RSL-RL backend only.

SB3 trains through `stable_baselines3.common.vec_env.VecEnv` (numpy, CPU), so the
SB3 wrapper copies observations to host memory every step — a known cost of
pairing SB3 with GeneLab's GPU-vectorized env. Hindsight Experience Replay is
available for the off-policy algorithms via `Sb3AgentCfg.her`, which exposes a
goal-conditioned observation and trains through SB3's `HerReplayBuffer`.

```bash
# An Sb3AgentCfg routes through the SB3 backend; the Franka pick-and-place task
# is SAC + HER + lift bonus + FSM demo prefill (see its example page).
GENELAB_SB3_DEMO_PATH=/tmp/franka_pp_demos.npz \
  uv run genelab train GeneLab-Franka-Pick-And-Place-v0 \
  --gpu --num-envs 32 --max-iterations 2000000
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
