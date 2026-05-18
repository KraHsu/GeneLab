# RL Runner

`genelab.rl` connects registered tasks to RSL-RL. It is deliberately thin: GeneLab owns task
resolution, config mutation, VecEnv adaptation, logs, profiling hooks, and distributed launch
helpers; RSL-RL owns the learning algorithm.

## Training flow

```text
TASKS.get(task_id)
└── TaskCfg.env + TaskCfg.agent
    └── ManagerBasedRlEnv
        └── RslRlVecEnvWrapper
            └── rsl_rl OnPolicyRunner.learn()
```

The main process writes `params/env.json`, `params/agent.json`, TensorBoard events, profiler traces,
and checkpoints.

## Playback flow

`play_task` prefers `TaskCfg.play_env` when present. It selects a policy source:

| Agent | Source |
|---|---|
| `zero` | Returns zero actions. |
| `random` | Uniform random actions. |
| `trained` | Loads a checkpoint and calls RSL-RL inference policy. |

Playback exits cleanly when the Genesis viewer closes or when `max_steps` is reached.

## Distributed training

`genelab train TASK --gpus N` relaunches the current command under `torchrun`. The parent computes a
shared log directory so every rank writes into the same run. `--num_envs` is total across ranks;
`--num_envs_per_gpu` is per rank.

## Where to continue

- [Run RL Experiments](../best-practices/rl-experiments.md)
- [Play and Train](../cli/play-train.md)
- [API Reference](../api/reference.md)
