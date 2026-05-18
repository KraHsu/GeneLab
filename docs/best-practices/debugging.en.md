# How to Debug Common Failures

This guide lists the shortest checks for common GeneLab failures.

## Unknown task

```bash
uv run genelab list tasks
uv run genelab --import my_project.tasks list tasks
```

If the second command works, the package is importable only when you pass `--import`. Install an
entry point or keep using an explicit import. If neither works, confirm the package is installed or
its `src/` directory is on `PYTHONPATH`.

## Unknown override path

```bash
uv run genelab info TASK_ID
```

Copy the path from `Overridable cfg paths`. Remember that `play` shortcuts may retarget
`env.simulation.*` to `play_env.simulation.*` when a task has `play_env`.

## Torch or Genesis import errors

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
uv run python -c "import genesis; print(genesis.__version__)"
```

Pick exactly one `torch-*` extra and resync if needed:

```bash
uv sync --reinstall-package torch --extra torch-cu128
```

## Cache or write-location errors

```bash
uv run genelab cache
```

This creates project-local cache directories and points `XDG_CACHE_HOME` and `MPLCONFIGDIR` at
`.cache/`.

## Viewer failures

First run headless:

```bash
uv run genelab play TASK_ID --steps 32
```

Then try viewer mode:

```bash
uv run genelab play TASK_ID --vis --steps 128
```

Viewer failures are usually graphics-driver, display, or OpenGL platform problems rather than
registry problems.

## Observation or reward shape errors

Manager terms should return batch-shaped tensors:

| Term type | Expected shape |
|---|---|
| observation term | `(num_envs, d)` or `(num_envs,)` |
| reward term | `(num_envs,)` |
| termination term | `(num_envs,)` bool |

Avoid scalar `()` tensors. They can broadcast incorrectly or fail later in the runner.

## Distributed env-count errors

`--num_envs` is total across ranks and must divide evenly by `--gpus`.

```bash
uv run genelab train TASK_ID --gpus 4 --num_envs 4096
```

Use `--num_envs_per_gpu` to avoid division:

```bash
uv run genelab train TASK_ID --gpus 4 --num_envs_per_gpu 1024
```
