# 如何运行 RL 实验

这份指南说明用 `genelab.rl` 训练和回放 GeneLab task 的实用流程。

## 1. 训练前先做 smoke test

任何长任务前先跑：

```bash
uv run genelab info TASK_ID
uv run genelab play TASK_ID --agent zero --steps 32
uv run genelab play TASK_ID --agent random --steps 64
uv run genelab train TASK_ID --num_envs 64 --max_iterations 2
```

这会检查注册表加载、env 构造、action 维度、observation group、reward、termination 和 runner 接线。

## 2. 为回放选择策略来源

| Agent | 行为 | 用途 |
|---|---|---|
| `zero` | 发送零动作。 | 被动物理和 reset 健康检查。 |
| `random` | 发送 `[-1, 1]` 均匀随机动作。 | action 边界与稳定性检查。 |
| `trained` | 通过 task runner 加载 checkpoint。 | 检查训练后行为。 |

checkpoint 回放：

```bash
uv run genelab play TASK_ID \
  --agent trained \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_300.pt
```

传 `--checkpoint` 时，`--agent` 默认就是 `trained`。

## 3. 明确控制 env 数量

单进程：

```bash
uv run genelab train TASK_ID --num_envs 4096
```

分布式：

```bash
uv run genelab train TASK_ID --gpus 4 --num_envs 4096
```

`--gpus 4` 时，`--num_envs 4096` 表示总数 4096，每个 rank 1024。想直接指定每个 rank
数量时用 `--num_envs_per_gpu`。

## 4. 保持日志可复现

在 `RslRlOnPolicyRunnerCfg` 中使用有意义的 `experiment_name` 和 `run_name`。GeneLab 写入：

```text
logs/rsl_rl/<experiment>/<timestamp-or-run>/
├── params/env.json
├── params/agent.json
└── model_*.pt
```

只有需要精确输出目录时才使用 `--log_dir PATH`，例如分布式重启或脚本化对比。

## 5. 先 profile 短 run

```bash
uv run genelab train TASK_ID \
  --prof \
  --prof-active 3 \
  --prof-repeat 1 \
  --max_iterations 10
uv run genelab prof open logs/torch_profile
```

profiler trace 增长很快。先用小 active 窗口确认体量，再扩大。

## 预期结果

每个长训练都应该有通过的 smoke test、明确的 env 数语义、可检查的 `params/*.json`，以及已知的 checkpoint 回放命令。
