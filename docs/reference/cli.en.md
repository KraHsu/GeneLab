# CLI Reference

GeneLab exposes one console script:

```bash
uv run genelab [global options] <command> [command options]
```

## Global options

| Option | Description |
|---|---|
| `--version` | Print GeneLab version and exit. |
| `--import MODULE` | Import an extension module before dispatch. Repeatable. |
| `--no-entry-points` | Skip installed `genelab.extensions` entry points. Useful for reproducible debugging with explicit imports. |
| `--help` | Show help. |

## Commands

| Command | Purpose |
|---|---|
| `cache` | Create project-local cache directories. |
| `list robots` | List registered robots. |
| `list envs` | List registered environments. |
| `list tasks` | List registered tasks. |
| `info NAME` | Show detail for a registered task, env, or robot. |
| `play TASK ...` | Run a registered task. |
| `train TASK ...` | Train a task with a supported runner. |
| `prof open [DIR]` | Open TensorBoard for profiler traces. |
| `project new NAME` | Scaffold an external GeneLab extension project. |

## Runtime flags

These flags are accepted after the task id for `play` and `train`.

| Flag | Description |
|---|---|
| `--vis`, `-v` | Enable the Genesis viewer. |
| `--gpu` | Use the Genesis GPU backend. |
| `--steps N` | In `play`, rollout steps. In `train`, short form for `--max_iterations N`. |
| `--dt X` | Override simulator timestep. |
| `--a.b.c VALUE` | Apply a dotted config override. |

## Runner flags

| Flag | Command | Description |
|---|---|---|
| `--num_envs N` | play/train | Total number of envs. In distributed train, divided by `--gpus`. |
| `--num_envs_per_gpu N` | play/train | Per-rank env count. Mutually exclusive with `--num_envs`. |
| `--agent zero|random|trained` | play | Select policy source. |
| `--checkpoint PATH` | play/train | Load a checkpoint. In play, defaults agent to `trained`. |
| `--seed N` | train | Override env and agent seed. |
| `--log_dir PATH` | train | Use a resolved log directory. |
| `--max_iterations N` | train | Override PPO iteration count. |
| `--gpus N` | train | Relaunch under `torchrun` with `N` local ranks. |

## Profiler flags

| Flag | Environment fallback | Default |
|---|---|---|
| `--prof` | `GENELAB_PROFILE=1` | off |
| `--prof-out PATH` | `GENELAB_PROFILE_OUT` | `logs/torch_profile` |
| `--prof-wait N` | `GENELAB_PROFILE_WAIT` | `10` |
| `--prof-warmup N` | `GENELAB_PROFILE_WARMUP` | `5` |
| `--prof-active N` | `GENELAB_PROFILE_ACTIVE` | `10` |
| `--prof-repeat N` | `GENELAB_PROFILE_REPEAT` | `2` |
| `--prof-record-shapes` | `GENELAB_PROFILE_RECORD_SHAPES=1` | off |
| `--prof-with-stack` | `GENELAB_PROFILE_WITH_STACK=1` | off |

## Completion and interactive recovery

Typer provides `--install-completion` and `--show-completion`. Completion loads installed entry
points but cannot see temporary `--import MODULE` extensions.

When stdin is a TTY, GeneLab can prompt for missing task ids, unknown names, invalid `--agent`
values, and unknown override paths. In scripts and CI, errors are raised directly.
