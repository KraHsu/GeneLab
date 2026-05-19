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

## RL 后端

训练后端由 task 的 agent 配置类型自动选择，无需任何 flag：

| Agent 配置 | 后端 | 算法 |
|---|---|---|
| `RslRlOnPolicyRunnerCfg` | `rsl_rl`（默认） | PPO |
| `SkrlAgentCfg` | `skrl` | PPO、A2C、SAC、TD3、DDPG |

[skrl](https://skrl.readthedocs.io) 后端为可选项——通过 `skrl` extra 安装（本仓库
`uv sync` 已包含；下游用户执行 `pip install genelab[skrl]`）。算法通过
`SkrlAgentCfg.algorithm` 选择。

skrl 以环境 **timestep**（而非 learning iteration）计量训练量，因此对 skrl task，
`--max_iterations N` 设定的是 skrl 的 timestep 预算。多 GPU（`--gpus`）仅
RSL-RL 后端支持。

```bash
# 注册为 SkrlAgentCfg 的 task 会走 skrl 后端。
uv run genelab train GeneLab-Franka-Pick-And-Place-skrl-v0 --num_envs 2048 --max_iterations 12000
```

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
