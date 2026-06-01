# 倒立摆

倒立摆示例是推荐的第一个可运行任务。它足够小，适合 smoke test，同时覆盖完整的 GeneLab train/play 路径。

## 任务

| Task id | 描述 |
|---|---|
| `GeneLab-Inverted-Pendulum-v0` | 单杆平衡（rsl_rl PPO）。 |
| `GeneLab-Double-Inverted-Pendulum-v0` | 双连杆平衡（rsl_rl PPO）。 |
| `GeneLab-Inverted-Pendulum-Skrl-v0` | 单杆平衡 —— 同一个 env，**skrl** PPO 后端。 |
| `GeneLab-Inverted-Pendulum-Memory-Rnn-v0` | "记住目标" —— 需要记忆的任务,用 **LSTM** 策略。 |
| `GeneLab-Inverted-Pendulum-Memory-Mlp-v0` | 同一任务的普通 **MLP** 基线 —— 结构上无法记忆。 |

## 安装并列出

```bash
uv pip install -e examples/inverted_pendulum
genelab list tasks
```

不安装时：

```bash
PYTHONPATH=examples/inverted_pendulum/src \
  genelab --import genelab_inverted_pendulum.tasks list tasks
```

## 运行

```bash
genelab play GeneLab-Inverted-Pendulum-v0 --steps 64
genelab play GeneLab-Inverted-Pendulum-v0 --vis --steps 500
genelab train GeneLab-Inverted-Pendulum-v0 --num_envs 64 --max_iterations 2

# 同一个 env，skrl PPO 后端（仅由 agent cfg 类型选择）：
genelab train GeneLab-Inverted-Pendulum-Skrl-v0 --num_envs 64 --max_iterations 4800
# skrl 的 checkpoint 命名为 agent_<timesteps>.pt，位于该 run 的 checkpoints/ 目录下：
genelab eval GeneLab-Inverted-Pendulum-Skrl-v0 logs/skrl/inverted_pendulum_skrl/<run>/checkpoints/agent_<N>.pt
```

> skrl 任务的 `genelab train` 需要安装可选依赖 `skrl`；注册与列出任务则不需要。

## 循环网络（RNN）记忆 showcase

`GeneLab-Inverted-Pendulum-Memory-Rnn-v0` 展示**循环网络真正有用的场景**。这是一个"记住目标"
任务:每个 episode 随机一个小车目标,但它只在**最初 5 步出现在观测里**,之后被抹掉。这一闪太短,
来不及在可见时把车开过去,所以策略必须**记住**目标、之后再去。无记忆的 MLP 在线索消失后无法
恢复目标,只能退回中点;而 LSTM 把它存进 hidden state,直接把车开到目标。(速度仍可观测,所以
唯一需要记忆的就是目标。)

这才是循环网络决定性占优的任务类别 —— 与单纯的*平衡*不同:平衡即使隐藏速度或负载,前馈 MLP
也能解,因为反馈镇定对未观测状态天生鲁棒。**RNN 买到的是"记忆",而不仅仅是"部分可观测"。**

训练两者,对比 post-cue 跟踪误差(到隐藏目标的平均距离):

```bash
genelab train GeneLab-Inverted-Pendulum-Memory-Rnn-v0 --num_envs 2048 --max_iterations 400
genelab train GeneLab-Inverted-Pendulum-Memory-Mlp-v0 --num_envs 2048 --max_iterations 400
```

| 策略 | 任务 | post-cue 跟踪误差(平均 \|cart − target\|) |
|---|---|---|
| **LSTM（循环）** | `...-Memory-Rnn-v0` | **≈ 0.013** —— 够到并保持隐藏目标 |
| MLP（基线） | `...-Memory-Mlp-v0` | ≈ 0.78 —— 回忆不出,卡在中点附近 |

(目标在 `[-0.8, 0.8]` 均匀采样;忽略目标的策略误差 ≈ 目标分布的幅度。)

!!! note "为什么这次训得动(之前的记忆尝试不行)"

    循环 PPO 用长度为 `num_steps_per_env` 的 truncated BPTT。记忆 cfg 把它设成 `100`(整整一个
    ~100 步 episode),于是每个 BPTT 窗口覆盖完整的"闪现→够到"依赖,梯度能从回忆一路传回编码线索
    那一步。用默认的 24 步窗口,LSTM 永远学不会把两者关联起来。

## 代码入口

| 文件 | 作用 |
|---|---|
| `tasks.py` | 注册 robot、env、task。 |
| `single/env_cfg.py` | 单杆 manager-based env 配置。 |
| `single/memory_env_cfg.py` | "记住目标" env(带遮罩的目标线索 + 跟踪奖励)。 |
| `single/rnn_cfg.py` | 记忆任务的 LSTM + MLP PPO 配置。 |
| `double/env_cfg.py` | 双杆 manager-based env 配置。 |
| `mdp.py` | 示例自己的 reward / termination / 记忆任务项。 |

## 另见

- [教程](../tutorial.md)
- [设计任务](../best-practices/task-design.md)
