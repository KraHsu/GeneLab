# Quickstart

Use this page when the environment is already installed and you want the shortest command path.
Commands are written as a bare `genelab`, which assumes the venv is active (`source .venv/bin/activate`);
see [Installation → Run commands](installation.md#run-commands) for the why and for the `uv run --no-sync` alternative.

## 1. List registered content

```bash
genelab list robots
genelab list envs
genelab list tasks
```

If a task is missing, load its extension explicitly:

```bash
PYTHONPATH=examples/inverted_pendulum/src \
  genelab --import genelab_inverted_pendulum.tasks list tasks
```

## 2. Inspect a task

```bash
genelab info GeneLab-Inverted-Pendulum-v0
```

Copy override paths from the printed config tree instead of guessing blindly.

## 3. Play

```bash
genelab play GeneLab-Inverted-Pendulum-v0 --steps 64
genelab play GeneLab-Inverted-Pendulum-v0 --vis --steps 500
genelab play GeneLab-Inverted-Pendulum-v0 --agent random --steps 128
```

## 4. Train

```bash
genelab train GeneLab-Inverted-Pendulum-v0 \
  --num_envs 64 \
  --max_iterations 2
```

Longer run:

```bash
genelab train GeneLab-Inverted-Pendulum-v0 \
  --num_envs 4096 \
  --max_iterations 300
```

## 5. Scaffold a project

```bash
genelab project new my_robot_project
uv pip install -e my_robot_project
genelab list tasks
```

## See also

- [Tutorial](../tutorial.md)
- [CLI Reference](../reference/cli.md)
- [Run RL Experiments](../best-practices/rl-experiments.md)
