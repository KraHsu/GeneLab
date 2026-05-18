# Inverted Pendulum

The inverted pendulum example is the recommended first runnable task. It is small enough for smoke
tests but exercises the full GeneLab train/play path.

## Tasks

| Task id | Description |
|---|---|
| `GeneLab-Inverted-Pendulum-v0` | Single pole balancing. |
| `GeneLab-Double-Inverted-Pendulum-v0` | Two-link pole balancing. |

## Install and list

```bash
uv pip install -e examples/inverted_pendulum
uv run genelab list tasks
```

Without installation:

```bash
PYTHONPATH=examples/inverted_pendulum/src \
  uv run genelab --import genelab_inverted_pendulum.tasks list tasks
```

## Run

```bash
uv run genelab play GeneLab-Inverted-Pendulum-v0 --steps 64
uv run genelab play GeneLab-Inverted-Pendulum-v0 --vis --steps 500
uv run genelab train GeneLab-Inverted-Pendulum-v0 --num_envs 64 --max_iterations 2
```

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
