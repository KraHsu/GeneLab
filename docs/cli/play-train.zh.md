# 运行时：play 与 train

`play` 运行已注册任务。`train` 在 task 提供 agent 配置时，通过支持的 runner 训练任务。
训练后的 `eval` / `export` / `benchmark` 几条 runtime 子命令，复用 `train` 产出的同一个
checkpoint。

## Play

```bash
genelab play TASK_ID --steps 128
genelab play TASK_ID --vis --steps 500
genelab play TASK_ID --agent random --steps 128
```

策略来源：

| Agent | 行为 |
|---|---|
| `zero` | 零动作。未提供 checkpoint 时默认。 |
| `random` | `[-1, 1]` 均匀随机动作。 |
| `trained` | 加载 checkpoint 并使用 runner 的 inference policy。 |

策略相关选项（`--agent`、`--checkpoint`、`--num-envs`、`--prof*`）仅适用于 RL 任务，
即 play 环境配置为 `ManagerBasedRlEnvCfg` 的任务。对于配置继承自基类
`ManagerBasedEnvCfg` 的非 RL **场景回放示例**（如 `GeneLab-Rubiks-Play-v0`、
`GeneLab-Wuji-Hand-Playback-v0`），将运行其自带的回放逻辑；传入这些选项会打印
警告并被忽略。`--steps` / `--vis` / `--headless` / `--gpu` / `--dt` 以及点号配置
覆盖对两类任务都生效。

checkpoint 回放：

```bash
genelab play TASK_ID \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_300.pt
```

!!! note "无显示器服务器上的 trained 回放"
    可训练 task 的 play env 默认启用 Genesis viewer（`vis=play`），因此
    `play --agent trained` 会尝试开窗口，在无显示器的机器上会以
    `No display detected` 报错。传入 `--headless`（与 `--vis` 互斥）强制
    `env.simulation.vis=false`：

    ```bash
    genelab play TASK_ID --agent trained \
      --checkpoint <ckpt> --headless
    ```

    无头回放是有界的：没有窗口可关，它会在 `simulation.steps`（用 `--steps`
    设置，默认 240）步后停止，而不会一直运行。用 `--max-steps N` 覆盖。

## 简写标志

`play` 和 `train` 都会把以下简写改写为 `env.simulation.*` override：

| 简写 | 等价 override |
|---|---|
| `-v`、`--vis` | `env.simulation.vis=true` |
| `--headless` | `env.simulation.vis=false`（与 `--vis` 互斥） |
| `--gpu` | `env.simulation.gpu=true` |
| `--steps N` | play: `env.simulation.steps=N`；train: 等价于 `--max_iterations N` |
| `--dt SECONDS` | `env.simulation.dt=SECONDS` |
| `--a.b.c VALUE` | 任意 dotted cfg path |

## Train

```bash
genelab train TASK_ID --num_envs 4096 --max_iterations 300
```

分布式训练：

```bash
genelab train TASK_ID --gpus 4 --num_envs 4096
```

`--num_envs` 表示所有 rank 的总数，必须能被 `--gpus` 整除。每 rank 语义用
`--num_envs_per_gpu`（与 `--num_envs` 互斥）。多 GPU 仅 RSL-RL 后端支持，
首次进入会自动通过 `torchrun` 重启。

### 训练中评估

加 `--eval_every K` 即可每 K 次迭代跑一次确定性 rollout；如有改进，会把
`best_model.<ext>` 写入 `--log_dir`：

```bash
genelab train TASK_ID --eval_every 50 --eval_episodes 20
```

| 选项 | 含义（默认） |
|---|---|
| `--eval_every K` | 每 K 次迭代评估一次。 |
| `--eval_episodes N` | 每次评估的回合数（10）。 |
| `--eval_num_envs N` | 评估用的并行 env 数（与训练相同）。 |
| `--eval_seed N` | 评估的 RNG seed（0）。 |

### 多 seed train

`--seeds 1,2,3` 把当前 `train` 调用拆成多个独立子进程，每个 seed 一个：

```bash
genelab train TASK_ID --seeds 1,2,3,4 --parallel 2 --num_envs 4096
```

- `--parallel N` 限制并发数，默认 1（顺序跑）。
- 每个子进程会被注入 `--seed S` 和 `--log_dir <parent>/seed_<S>`。
- 不显式指定 `--log_dir` 时，父目录是
  `logs/multi-seed/<task_id>/<YYYY-MM-DD_HH-MM-SS>/`。
- 任一 seed 失败，命令以非零状态退出。

## RL 后端

训练后端由 task 的 agent 配置类型自动选择，无需任何标志：

| Agent 配置 | 后端 | 算法 |
|---|---|---|
| `RslRlOnPolicyRunnerCfg` | `rsl_rl`（默认） | PPO |
| `SkrlAgentCfg` | `skrl` | PPO、A2C、SAC、TD3、DDPG |
| `Sb3AgentCfg` | `sb3` | PPO、A2C、SAC、TD3、DDPG（含 HER） |

[skrl](https://skrl.readthedocs.io) 与
[Stable-Baselines3](https://stable-baselines3.readthedocs.io) 后端为可选项——
通过 `skrl` / `sb3` extra 安装（本仓库 `uv sync` 已包含两者；下游用户执行
`pip install genelab[skrl]` 或 `genelab[sb3]`）。算法通过
`SkrlAgentCfg.algorithm` / `Sb3AgentCfg.algorithm` 选择。

skrl 与 SB3 均以环境 **timestep**（而非 learning iteration）计量训练量，因此对
这两类 task，`--max_iterations N` 设定的是 timestep 预算。多 GPU（`--gpus`）仅
RSL-RL 后端支持。

SB3 通过 `stable_baselines3.common.vec_env.VecEnv`（numpy、CPU）训练，因此 SB3
wrapper 每步都会把观测拷贝到主机内存——这是 SB3 与 GeneLab GPU 向量化环境配合
的已知开销。Hindsight Experience Replay 通过 `Sb3AgentCfg.her` 为离策略算法提供：
它暴露目标条件化观测并经由 SB3 的 `HerReplayBuffer` 训练。

```bash
# 注册为 Sb3AgentCfg 的 task 会走 SB3 后端；Franka 抓取放置示例使用
# SAC + HER + lift bonus + FSM demo prefill 组合（详见示例页面）。
GENELAB_SB3_DEMO_PATH=/tmp/franka_pp_demos.npz \
  genelab train GeneLab-Franka-Pick-And-Place-v0 \
  --gpu --num-envs 32 --max-iterations 2000000
```

## 训练后的工具命令

`eval`、`export`、`benchmark` 都以已注册 task + checkpoint 作为入参，复用 task 的
play env 配置。

### Eval

确定性 rollout，写出 `eval.json`（包含 `return_mean`、`length_mean`，若 task 暴露
`extras['is_success']` 则附带 `success_rate`）：

```bash
genelab eval TASK_ID logs/.../model_300.pt \
  --num-envs 64 --episodes 100 --out eval.json
```

`--deterministic` / `--stochastic` 切换策略；`--max-steps` 设安全步上限。

### Export

把 policy 导出为 TorchScript 或 ONNX（per-term scale/clip 烘焙进模型）：

```bash
genelab export TASK_ID logs/.../model_300.pt --format onnx --out policy.onnx
```

同时写出 `<OUTPUT>.metadata.json`，记录 obs schema。

### Benchmark

按 JSON 套件批量 eval 并聚合参考数：

```bash
genelab benchmark --suite suite.json --out report.json
genelab benchmark --suite suite.json --reference baseline.json --tolerance 0.1
```

`suite.json` 是 `[{"task": ..., "checkpoint": ..., "episodes": ..., "seed": ..., "num_envs": ...}, ...]`。
传 `--reference` 时与 baseline 比 `return_mean`，下降超过 `--tolerance` 视为回归，
命令以非零状态退出——可直接作为 CI 回归门。

## 配置 override

task id 后的任何未知选项都会被当作 dotted config override：

```bash
genelab play TASK_ID \
  --env.simulation.dt 0.005 \
  --env.rewards_cfg.action_rate.weight -0.01
```

## 另见

- [运行 RL 实验](../best-practices/rl-experiments.md)
- [RL runner](../concepts/rl-runner.md)
- [Eval 与 export](../concepts/eval-and-export.md)
