# RL runner

`genelab.rl` 把已注册 task 接到 RSL-RL。它刻意保持很薄：GeneLab 负责 task 解析、配置修改、
VecEnv 适配、日志、profiling hook 和分布式启动 helper；RSL-RL 负责学习算法。

## 训练流程

```text
TASKS.get(task_id)
└── TaskCfg.env + TaskCfg.agent
    └── ManagerBasedRlEnv
        └── RslRlVecEnvWrapper
            └── rsl_rl OnPolicyRunner.learn()
```

main process 写入 `params/env.json`、`params/agent.json`、TensorBoard event、profiler trace 和 checkpoint。

## 回放流程

`play_task` 在存在 `TaskCfg.play_env` 时优先使用它。策略来源：

| Agent | 来源 |
|---|---|
| `zero` | 返回零动作。 |
| `random` | 均匀随机动作。 |
| `trained` | 加载 checkpoint，并调用 RSL-RL inference policy。 |

Genesis viewer 关闭或达到 `max_steps` 时，回放会干净退出。

## 分布式训练

`genelab train TASK --gpus N` 会把当前命令重新拉起到 `torchrun` 下。父进程预先计算共享日志目录，让每个 rank 写入同一个 run。`--num_envs` 是所有 rank 总数；`--num_envs_per_gpu` 是每 rank 数量。

## 继续阅读

- [运行 RL 实验](../best-practices/rl-experiments.md)
- [play 与 train](../cli/play-train.md)
- [API 参考](../api/reference.md)
