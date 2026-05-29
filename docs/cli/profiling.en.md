# Profiling

GeneLab forwards profiler flags to `torch.profiler` through `genelab.rl.maybe_profile`.

## Enable profiling

```bash
genelab train TASK_ID \
  --prof \
  --prof-active 3 \
  --prof-repeat 1 \
  --max_iterations 10
```

Open traces:

```bash
genelab prof open                       # defaults to logs/torch_profile
genelab prof open <log_dir>             # explicit directory
genelab prof open <log_dir> --port 6007 # custom TensorBoard port
genelab prof open <log_dir> --host 0.0.0.0
```

## Flags

| Flag | Env-var equivalent | Meaning (default) |
|---|---|---|
| `--prof` | `GENELAB_PROFILE` | Enable profiling. |
| `--prof-out PATH` | `GENELAB_PROFILE_OUT` | TensorBoard trace directory. |
| `--prof-wait N` | `GENELAB_PROFILE_WAIT` | Initial wait steps (10). |
| `--prof-warmup N` | `GENELAB_PROFILE_WARMUP` | Warmup steps before recording (5). |
| `--prof-active N` | `GENELAB_PROFILE_ACTIVE` | Recorded steps per cycle (10). |
| `--prof-repeat N` | `GENELAB_PROFILE_REPEAT` | Number of cycles (2). |
| `--prof-record-shapes` | `GENELAB_PROFILE_RECORD_SHAPES` | Record tensor shapes. |
| `--prof-with-stack` | `GENELAB_PROFILE_WITH_STACK` | Capture Python stacks. Higher overhead. |

CLI flags always take precedence over the env vars.

## Practical defaults

Start short. A profiler step is advanced once per env step in GeneLab's RL loop, so traces can grow
quickly on large vectorized runs.

## Distributed runs

Only the main process writes profiler traces. Keep `--prof-active` small when profiling distributed
training.

## `prof open` prerequisites

- `tensorboard` must be on `PATH` (`uv pip install tensorboard`, or via this checkout's `rl` extra).
- The directory you pass must exist; an empty directory is fine.

## See also

- [Run RL Experiments](../best-practices/rl-experiments.md)
- [CLI Reference](../reference/cli.md)
