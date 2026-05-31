# RL runner

`genelab.rl` 把已注册 task 接到可插拔的 RL **后端**。它刻意保持很薄：GeneLab 负责 task 解析、
配置修改、环境构建、bridge 生命周期、日志、profiling hook 和分布式启动 helper；后端负责学习算法。

## 后端

`train_task` / `play_task` 是与后端无关的分发器。后端由 `TaskCfg.agent` 的类型决定：

| Agent 配置 | 后端 | 算法 |
|---|---|---|
| `RslRlOnPolicyRunnerCfg` | `rsl_rl`（默认） | PPO |
| `SkrlAgentCfg` | `skrl` | PPO、A2C、SAC、TD3、DDPG |
| `Sb3AgentCfg` | `sb3` | PPO、A2C、SAC、TD3、DDPG（含 HER） |

后端位于 `genelab.rl.backends`，按配置类型自行注册，`select_backend(agent_cfg)` 通过
`type[BackendConfig]` 索引的 typed registry 解析。接入另一个库只需新增一个 `Backend`
（`train` / `play`）及其继承 `genelab.rl.config.BackendConfig` 的 agent 配置 dataclass——分发器和
CLI 无需改动。

## 训练流程

```text
TASKS.get(task_id)
└── TaskCfg.env + TaskCfg.agent
    └── ManagerBasedRlEnv
        └── select_backend(agent_cfg).train(TrainContext)
            ├── rsl_rl:  RslRlVecEnvWrapper  → OnPolicyRunner.learn()
            ├── skrl:    GenelabSkrlWrapper  → SequentialTrainer.train()
            └── sb3:     GenelabSb3VecEnv    → model.learn()
```

main process 写入 `params/env.json`、`params/agent.json`、TensorBoard event、profiler trace 和
checkpoint。RSL-RL 写到 `logs/rsl_rl/`，skrl 写到 `logs/skrl/`，SB3 写到 `logs/sb3/`。

## 回放流程

`play_task` 在存在 `TaskCfg.play_env` 时优先使用它。策略来源：

| Agent | 来源 |
|---|---|
| `zero` | 返回零动作。 |
| `random` | 均匀随机动作。 |
| `trained` | 加载 checkpoint，并调用后端的 inference policy。 |

回放时长取决于是否开启 viewer，而非 agent 类型：开启 viewer（`vis=true`）时一直运行到窗口关闭；无头（`vis=false`）时对所有 agent 类型都在 `simulation.steps`（由 `--steps` 设置）步后停止，因此无头运行不会卡死。显式的 `max_steps`（`--max-steps`）会覆盖两者。

## 分布式训练

`genelab train TASK --gpus N` 会把当前命令重新拉起到 `torchrun` 下。父进程预先计算共享日志目录，让每个 rank 写入同一个 run。`--num_envs` 是所有 rank 总数；`--num_envs_per_gpu` 是每 rank 数量。

## 继续阅读

- [运行 RL 实验](../best-practices/rl-experiments.md)
- [play 与 train](../cli/play-train.md)
- [API 参考](../api/reference.md)
