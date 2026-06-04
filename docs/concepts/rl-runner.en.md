# RL Runner

`genelab.rl` connects registered tasks to a pluggable RL **backend**. It is deliberately thin:
GeneLab owns task resolution, config mutation, env construction, the bridge lifecycle, logs,
profiling hooks, and distributed launch helpers; the backend owns the learning algorithm.

## Backends

`train_task` / `play_task` are backend-agnostic dispatchers. The backend is chosen from the type of
`TaskCfg.agent`:

| Agent config | Backend | Algorithms |
|---|---|---|
| `RslRlOnPolicyRunnerCfg` | `rsl_rl` (default) | PPO |
| `SkrlAgentCfg` | `skrl` | PPO, A2C, SAC, TD3, DDPG |
| `Sb3AgentCfg` | `sb3` | PPO, A2C, SAC, TD3, DDPG (+ HER) |

Backends live under `genelab.rl.backends` and register themselves by config type;
`select_backend(agent_cfg)` resolves one through a typed registry keyed by
`type[BackendConfig]`. Adding another library means adding a `Backend` (`train` / `play`) plus an
agent-config dataclass that subclasses `genelab.rl.config.BackendConfig` — no change to the
dispatcher or CLI.

## Training flow

```text
TASKS.get(task_id)
└── TaskCfg.env + TaskCfg.agent
    └── ManagerBasedRlEnv
        └── select_backend(agent_cfg).train(TrainContext)
            ├── rsl_rl:  RslRlVecEnvWrapper  → OnPolicyRunner.learn()
            ├── skrl:    GenelabSkrlWrapper  → SequentialTrainer.train()
            └── sb3:     GenelabSb3VecEnv    → model.learn()
```

The main process writes `params/env.json`, `params/agent.json`, TensorBoard events, profiler traces,
and checkpoints. RSL-RL logs under `logs/rsl_rl/`, skrl under `logs/skrl/`, SB3 under `logs/sb3/`.

## Playback flow

`play_task` prefers `TaskCfg.play_env` when present. It selects a policy source:

| Agent | Source |
|---|---|
| `zero` | Returns zero actions. |
| `random` | Uniform random actions. |
| `trained` | Loads a checkpoint and calls the backend's inference policy. |

Playback length is gated on the viewer, not the agent kind: with a viewer
(`vis=true`) it runs until the window is closed; headless (`vis=false`) it stops
after `simulation.steps` steps (what `--steps` sets) for every agent kind, so a
headless run can't hang. An explicit `max_steps` (`--max-steps`) overrides both.

## Distributed training

`genelab train TASK --gpus N` relaunches the current command under `torchrun`. The parent computes a
shared log directory so every rank writes into the same run. `--num_envs` is total across ranks;
`--num_envs_per_gpu` is per rank.

## Where to continue

- [Run RL Experiments](../best-practices/rl-experiments.md)
- [Play and Train](../cli/play-train.md)
- [API Reference](../api/reference.md)
