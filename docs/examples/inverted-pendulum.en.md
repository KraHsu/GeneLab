# Inverted Pendulum

The inverted pendulum example is the recommended first runnable task. It is small enough for smoke
tests but exercises the full GeneLab train/play path.

## Tasks

| Task id | Description |
|---|---|
| `GeneLab-Inverted-Pendulum-v0` | Single pole balancing (rsl_rl PPO). |
| `GeneLab-Double-Inverted-Pendulum-v0` | Two-link pole balancing (rsl_rl PPO). |
| `GeneLab-Inverted-Pendulum-Skrl-v0` | Single pole balancing — same env, **skrl** PPO backend. |

## Install and list

```bash
uv pip install -e examples/inverted_pendulum
genelab list tasks
```

Without installation:

```bash
PYTHONPATH=examples/inverted_pendulum/src \
  genelab --import genelab_inverted_pendulum.tasks list tasks
```

## Run

```bash
genelab play GeneLab-Inverted-Pendulum-v0 --steps 64
genelab play GeneLab-Inverted-Pendulum-v0 --vis --steps 500
genelab train GeneLab-Inverted-Pendulum-v0 --num_envs 64 --max_iterations 2

# Same env, skrl PPO backend (selected purely by the agent cfg type):
genelab train GeneLab-Inverted-Pendulum-Skrl-v0 --num_envs 64 --max_iterations 4800
# skrl names checkpoints agent_<timesteps>.pt under the run's checkpoints/ dir:
genelab eval GeneLab-Inverted-Pendulum-Skrl-v0 logs/skrl/inverted_pendulum_skrl/<run>/checkpoints/agent_<N>.pt
```

> `genelab train` for the skrl task needs the optional `skrl` dependency
> installed; registering and listing the task does not.

## Code entry points

| File | Role |
|---|---|
| `tasks.py` | Registers robots, envs, and tasks. |
| `single/env_cfg.py` | Single-pole manager-based env config. |
| `double/env_cfg.py` | Double-pole manager-based env config. |
| `mdp.py` | Example-specific reward and termination helpers. |

## See also

- [Tutorial](../tutorial.md)
- [Task design](../best-practices/task-design.md)
