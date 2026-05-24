# 倒立摆

倒立摆示例是推荐的第一个可运行任务。它足够小，适合 smoke test，同时覆盖完整的 GeneLab train/play 路径。

## 任务

| Task id | 描述 |
|---|---|
| `GeneLab-Inverted-Pendulum-v0` | 单杆平衡（rsl_rl PPO）。 |
| `GeneLab-Double-Inverted-Pendulum-v0` | 双连杆平衡（rsl_rl PPO）。 |
| `GeneLab-Inverted-Pendulum-Skrl-v0` | 单杆平衡 —— 同一个 env，**skrl** PPO 后端。 |

## 安装并列出

```bash
uv pip install -e examples/inverted_pendulum
uv run genelab list tasks
```

不安装时：

```bash
PYTHONPATH=examples/inverted_pendulum/src \
  uv run genelab --import genelab_inverted_pendulum.tasks list tasks
```

## 运行

```bash
uv run genelab play GeneLab-Inverted-Pendulum-v0 --steps 64
uv run genelab play GeneLab-Inverted-Pendulum-v0 --vis --steps 500
uv run genelab train GeneLab-Inverted-Pendulum-v0 --num_envs 64 --max_iterations 2

# 同一个 env，skrl PPO 后端（仅由 agent cfg 类型选择）：
uv run genelab train GeneLab-Inverted-Pendulum-Skrl-v0 --num_envs 64 --max_iterations 4800
# skrl 的 checkpoint 命名为 agent_<timesteps>.pt，位于该 run 的 checkpoints/ 目录下：
uv run genelab eval GeneLab-Inverted-Pendulum-Skrl-v0 logs/skrl/inverted_pendulum_skrl/<run>/checkpoints/agent_<N>.pt
```

> skrl 任务的 `genelab train` 需要安装可选依赖 `skrl`；注册与列出任务则不需要。

## 代码入口

| 文件 | 作用 |
|---|---|
| `tasks.py` | 注册 robot、env、task。 |
| `single/env_cfg.py` | 单杆 manager-based env 配置。 |
| `double/env_cfg.py` | 双杆 manager-based env 配置。 |
| `mdp.py` | 示例自己的 reward 与 termination helper。 |

## 另见

- [教程](../tutorial.md)
- [设计任务](../best-practices/task-design.md)
