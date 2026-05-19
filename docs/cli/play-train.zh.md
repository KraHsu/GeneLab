# play 与 train

`play` 运行已注册任务。`train` 在 task 提供 agent 配置时，通过支持的 runner 训练任务。

## Play

```bash
uv run genelab play TASK_ID --steps 128
uv run genelab play TASK_ID --vis --steps 500
uv run genelab play TASK_ID --agent random --steps 128
```

策略来源：

| Agent | 行为 |
|---|---|
| `zero` | 零动作。未提供 checkpoint 时默认。 |
| `random` | `[-1, 1]` 均匀随机动作。 |
| `trained` | 加载 checkpoint 并使用 runner inference policy。 |

checkpoint 回放：

```bash
uv run genelab play TASK_ID \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_300.pt
```

## Train

```bash
uv run genelab train TASK_ID --num_envs 4096 --max_iterations 300
```

分布式训练：

```bash
uv run genelab train TASK_ID --gpus 4 --num_envs 4096
```

`--num_envs` 表示所有 rank 的总数，必须能被 `--gpus` 整除。每 rank 语义用
`--num_envs_per_gpu`。

## 配置 override

task id 后的任何未知选项都会被当作 dotted config override：

```bash
uv run genelab play TASK_ID \
  --env.simulation.dt 0.005 \
  --env.rewards_cfg.action_rate.weight -0.01
```

## 另见

- [运行 RL 实验](../best-practices/rl-experiments.md)
- [CLI 参考](../reference/cli.md)
- [RL runner](../concepts/rl-runner.md)
