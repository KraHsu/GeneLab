# Runtime: play and train

`play` runs a registered task. `train` runs a registered task through a supported runner when the
task provides an agent config. The post-training runtime subcommands —
`eval`, `export`, and `benchmark` — take the same checkpoint produced by `train`.

## Play

```bash
genelab play TASK_ID --steps 128
genelab play TASK_ID --vis --steps 500
genelab play TASK_ID --agent random --steps 128
```

Policy sources:

| Agent | Behavior |
|---|---|
| `zero` | Zero actions. Default when no checkpoint is given. |
| `random` | Uniform random actions in `[-1, 1]`. |
| `trained` | Load a checkpoint and use the runner inference policy. |

The policy options (`--agent`, `--checkpoint`, `--num-envs`, `--prof*`) apply only to RL
tasks — those whose play env config is a `ManagerBasedRlEnvCfg`. Non-RL **scene-playback
demos** (e.g. `GeneLab-Rubiks-Play-v0`, `GeneLab-Wuji-Hand-Playback-v0`), whose config
subclasses the base `ManagerBasedEnvCfg`, run their own built-in playback; passing those
options prints a warning and they are ignored. `--steps` / `--vis` / `--headless` / `--gpu` /
`--dt` and dotted config overrides still apply to both.

Checkpoint replay:

```bash
genelab play TASK_ID \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_300.pt
```

!!! note "Trained playback on a headless server"
    Trainable tasks enable the Genesis viewer in their play env (`vis=play`), so
    `play --agent trained` opens a window by default and aborts with
    `No display detected` on a display-less machine. Pass `--headless` (mutually
    exclusive with `--vis`) to force `env.simulation.vis=false`:

    ```bash
    genelab play TASK_ID --agent trained \
      --checkpoint <ckpt> --headless
    ```

## Shortcut flags

Both `play` and `train` rewrite the following shortcuts into `env.simulation.*` overrides:

| Shortcut | Override |
|---|---|
| `-v`, `--vis` | `env.simulation.vis=true` |
| `--headless` | `env.simulation.vis=false` (mutually exclusive with `--vis`) |
| `--gpu` | `env.simulation.gpu=true` |
| `--steps N` | play: `env.simulation.steps=N`; train: alias for `--max_iterations N` |
| `--dt SECONDS` | `env.simulation.dt=SECONDS` |
| `--a.b.c VALUE` | Any dotted cfg path |

## Train

```bash
genelab train TASK_ID --num_envs 4096 --max_iterations 300
```

For distributed training:

```bash
genelab train TASK_ID --gpus 4 --num_envs 4096
```

`--num_envs` is total across ranks and must divide evenly by `--gpus`. Use
`--num_envs_per_gpu` for per-rank semantics (mutually exclusive with `--num_envs`).
Multi-GPU is RSL-RL only; the first entry automatically relaunches under `torchrun`.

### In-training eval

Pass `--eval_every K` to run a deterministic rollout every K iterations. On improvement, the
runner writes `best_model.<ext>` into `--log_dir`:

```bash
genelab train TASK_ID --eval_every 50 --eval_episodes 20
```

| Option | Meaning (default) |
|---|---|
| `--eval_every K` | Evaluate every K iterations. |
| `--eval_episodes N` | Episodes per evaluation (10). |
| `--eval_num_envs N` | Parallel envs during eval (matches training). |
| `--eval_seed N` | RNG seed for the eval rollout (0). |

### Multi-seed train

`--seeds 1,2,3` fans out the current `train` invocation into one independent subprocess per seed:

```bash
genelab train TASK_ID --seeds 1,2,3,4 --parallel 2 --num_envs 4096
```

- `--parallel N` caps concurrency (default 1 — sequential).
- Each child is invoked with `--seed S` and `--log_dir <parent>/seed_<S>`.
- Without an explicit `--log_dir`, the parent is
  `logs/multi-seed/<task_id>/<YYYY-MM-DD_HH-MM-SS>/`.
- If any seed fails, the command exits non-zero.

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
  genelab train GeneLab-Franka-Pick-And-Place-v0 \
  --gpu --num-envs 32 --max-iterations 2000000
```

## Post-training subcommands

`eval`, `export`, and `benchmark` all take a registered task plus a checkpoint and reuse the
task's play env config.

### Eval

Deterministic rollout that writes `eval.json` (`return_mean`, `length_mean`, and
`success_rate` if the task publishes `extras['is_success']`):

```bash
genelab eval TASK_ID logs/.../model_300.pt \
  --num-envs 64 --episodes 100 --out eval.json
```

`--deterministic` / `--stochastic` toggles the policy mode; `--max-steps` caps the rollout.

### Export

Export the policy as TorchScript or ONNX (per-term scale/clip baked into the model):

```bash
genelab export TASK_ID logs/.../model_300.pt --format onnx --out policy.onnx
```

A sibling `<OUTPUT>.metadata.json` records the observation schema.

### Benchmark

Batch eval driven by a JSON suite, aggregated into one report:

```bash
genelab benchmark --suite suite.json --out report.json
genelab benchmark --suite suite.json --reference baseline.json --tolerance 0.1
```

`suite.json` is `[{"task": ..., "checkpoint": ..., "episodes": ..., "seed": ..., "num_envs": ...}, ...]`.
With `--reference`, the command compares `return_mean` against the baseline and exits non-zero
when any task drops more than `--tolerance` — usable directly as a CI regression gate.

## Config overrides

Any unknown option after the task id is treated as a dotted config override:

```bash
genelab play TASK_ID \
  --env.simulation.dt 0.005 \
  --env.rewards_cfg.action_rate.weight -0.01
```

## See also

- [Run RL Experiments](../best-practices/rl-experiments.md)
- [RL runner](../concepts/rl-runner.md)
- [Eval and export](../concepts/eval-and-export.md)
