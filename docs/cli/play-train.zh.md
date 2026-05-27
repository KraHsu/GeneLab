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

策略相关选项（`--agent`、`--checkpoint`、`--num-envs`、`--prof*`）仅适用于 RL 任务，即 play
环境配置为 `ManagerBasedRlEnvCfg` 的任务。对于配置继承自基类 `ManagerBasedEnvCfg` 的非 RL
**场景回放示例**（如 `GeneLab-Rubiks-Play-v0`、`GeneLab-Wuji-Hand-Playback-v0`），将运行其
自带的回放逻辑；传入这些选项会打印告警并被忽略。`--steps` / `--vis` / `--gpu` 以及点号配置
覆盖对两类任务都生效。

checkpoint 回放：

```bash
uv run genelab play TASK_ID \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_300.pt
```

!!! note "无显示器服务器上的 trained 回放"
    可训练 task 的 play env 默认启用 Genesis viewer（`vis=play`），因此
    `play --agent trained` 会尝试开窗口，在无显示器的机器上会以
    `No display detected` 报错。传入 `--headless`（与 `--vis` 互斥）强制
    `env.simulation.vis=false`：

    ```bash
    uv run genelab play TASK_ID --agent trained \
      --checkpoint <ckpt> --headless
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
  uv run genelab train GeneLab-Franka-Pick-And-Place-v0 \
  --gpu --num-envs 32 --max-iterations 2000000
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
