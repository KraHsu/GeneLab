# CLI Reference

GeneLab exposes one console script:

```bash
genelab [global options] <command> [command options]
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
| `eval TASK CKPT ...` | Deterministic rollout of a checkpoint; writes `eval.json`. |
| `export TASK CKPT ...` | Export a checkpoint's policy to TorchScript or ONNX (scale/clip baked in). |
| `benchmark --suite ...` | Batch-eval a JSON suite into one report; regression gate with `--reference`. |
| `prof open [DIR]` | Open TensorBoard for profiler traces. |
| `project new NAME` | Scaffold an external GeneLab extension project. |
| `asset list` | List every asset declared under `genelab.asset_zoo` with cached / not-cached status and size. |
| `asset info NAME` | Show download URL, md5, filename, archive member, and cache path for one asset. |
| `asset download NAME` | Download a single asset (md5-verified, idempotent). Add `--all` for every asset, `--force` to re-fetch even if cached. |
| `asset purge NAME` | Remove one asset (or `--all`) from the local cache. Pass `--yes` to skip the confirmation prompt. |

The asset subcommands render the same Rich download progress bar every other CLI
surface uses (`play`, `train`, `eval`, `export`, `info`), so a first-touch
download from any command shows a progress bar instead of staying silent.

## Runtime flags

These flags are accepted after the task id for `play` and `train`.

| Flag | Description |
|---|---|
| `--vis`, `-v` | Enable the Genesis viewer. |
| `--headless` | Force no viewer (`env.simulation.vis=false`); mutually exclusive with `--vis`. Needed for `play --agent trained` on a display-less server, whose play env enables the viewer by default. |
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
| `--eval-every K` | train | Run a deterministic eval every K iterations; save `best_model` on improvement. |
| `--eval-episodes N` | train | Episodes per in-training eval (10). |
| `--eval-num-envs N` | train | Parallel envs during in-training eval (matches training). |
| `--eval-seed N` | train | RNG seed for the in-training eval rollout (0). |
| `--seeds 1,2,3` | train | Fan out into one subprocess per seed (multi-seed sweep). |
| `--parallel N` | train | Cap on concurrent seed subprocesses (1 — sequential). |

## Post-training flags

These flags follow the `TASK CHECKPOINT` positionals on `eval` / `export`, or stand alone on
`benchmark`.

| Flag | Command | Description |
|---|---|---|
| `--num-envs N` | eval | Parallel envs for the rollout. |
| `--episodes N` | eval | Minimum complete episodes to collect. |
| `--seed N` | eval | RNG seed (env-level only; policy is deterministic). |
| `--deterministic` / `--stochastic` | eval | Policy action mode (deterministic by default). |
| `--max-steps N` | eval | Safety cap on rollout steps. |
| `--out PATH` | eval/export/benchmark | Output path (`eval.json` / policy file / report JSON). |
| `--format torchscript\|onnx` | export | Serialization format (`torchscript` by default). |
| `--opset N` | export | ONNX opset version (ignored for torchscript). |
| `--suite PATH` | benchmark | Benchmark suite JSON (list of task/checkpoint entries). |
| `--reference PATH` | benchmark | Prior report JSON to compare `return_mean` against. |
| `--tolerance X` | benchmark | Max fractional `return_mean` drop before a regression (exits non-zero). |

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
