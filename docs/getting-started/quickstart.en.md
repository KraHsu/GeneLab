# Quickstart

Use this page when the environment is already installed and you want the shortest command path.

## 1. List registered content

```bash
uv run genelab list robots
uv run genelab list envs
uv run genelab list tasks
```

If a task is missing, load its extension explicitly:

```bash
PYTHONPATH=examples/inverted_pendulum/src \
  uv run genelab --import genelab_inverted_pendulum.tasks list tasks
```

## 2. Inspect a task

```bash
uv run genelab info GeneLab-Inverted-Pendulum-v0
```

Copy override paths from the printed config tree instead of guessing blindly.

## 3. Play

```bash
uv run genelab play GeneLab-Inverted-Pendulum-v0 --steps 64
uv run genelab play GeneLab-Inverted-Pendulum-v0 --vis --steps 500
uv run genelab play GeneLab-Inverted-Pendulum-v0 --agent random --steps 128
```

## 4. Train

```bash
uv run genelab train GeneLab-Inverted-Pendulum-v0 \
  --num_envs 64 \
  --max_iterations 2
```

Longer run:

```bash
uv run genelab train GeneLab-Inverted-Pendulum-v0 \
  --num_envs 4096 \
  --max_iterations 300
```

## 5. Scaffold a project

```bash
uv run genelab project new my_robot_project
uv pip install -e my_robot_project
uv run genelab list tasks
```

## See also

- [Tutorial](../tutorial.md)
- [CLI Reference](../reference/cli.md)
- [Run RL Experiments](../best-practices/rl-experiments.md)
