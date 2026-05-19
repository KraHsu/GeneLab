# How to Run RL Experiments

This guide describes the practical flow for training and replaying GeneLab tasks with `genelab.rl`.

## 1. Smoke-test before training

Run these before any long job:

```bash
uv run genelab info TASK_ID
uv run genelab play TASK_ID --agent zero --steps 32
uv run genelab play TASK_ID --agent random --steps 64
uv run genelab train TASK_ID --num_envs 64 --max_iterations 2
```

This checks registry loading, env construction, action dimensions, observation groups, rewards,
terminations, and runner wiring.

## 2. Pick the policy source for playback

| Agent | Behavior | Use case |
|---|---|---|
| `zero` | Sends zero actions. | Passive physics and reset health checks. |
| `random` | Sends uniform random actions in `[-1, 1]`. | Action bounds and stability checks. |
| `trained` | Loads a checkpoint through the task runner. | Inspect trained behavior. |

Checkpoint playback:

```bash
uv run genelab play TASK_ID \
  --agent trained \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_300.pt
```

Passing `--checkpoint` defaults `--agent` to `trained`.

## 3. Control env count deliberately

Single process:

```bash
uv run genelab train TASK_ID --num_envs 4096
```

Distributed:

```bash
uv run genelab train TASK_ID --gpus 4 --num_envs 4096
```

With `--gpus 4`, `--num_envs 4096` means 4096 total, 1024 per rank. Use
`--num_envs_per_gpu` when you want to specify the per-rank value directly.

## 4. Keep logs reproducible

Use meaningful `experiment_name` and `run_name` in `RslRlOnPolicyRunnerCfg`. GeneLab writes:

```text
logs/rsl_rl/<experiment>/<timestamp-or-run>/
├── params/env.json
├── params/agent.json
└── model_*.pt
```

Use `--log_dir PATH` only when you need an exact output directory, such as a distributed relaunch or
scripted comparison.

## 5. Profile short runs first

```bash
uv run genelab train TASK_ID \
  --prof \
  --prof-active 3 \
  --prof-repeat 1 \
  --max_iterations 10
uv run genelab prof open logs/torch_profile
```

Profiler traces grow quickly. Start with a small active window and increase only after confirming
the trace size is manageable.

## Expected result

Every long run should have a passing smoke test, explicit env count semantics, inspectable
`params/*.json`, and a known checkpoint replay command.
