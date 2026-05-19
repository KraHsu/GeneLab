# Profiling

GeneLab forwards profiler flags to `torch.profiler` through `genelab.rl.maybe_profile`.

## Enable profiling

```bash
uv run genelab train TASK_ID \
  --prof \
  --prof-active 3 \
  --prof-repeat 1 \
  --max_iterations 10
```

Open traces:

```bash
uv run genelab prof open logs/torch_profile
```

## Flags

| Flag | Meaning |
|---|---|
| `--prof` | Enable profiling. |
| `--prof-out PATH` | Trace output directory. |
| `--prof-wait N` | Initial wait steps. |
| `--prof-warmup N` | Warmup steps before recording. |
| `--prof-active N` | Recorded steps per cycle. |
| `--prof-repeat N` | Number of cycles. |
| `--prof-record-shapes` | Record tensor shapes. |
| `--prof-with-stack` | Capture Python stacks. Higher overhead. |

## Practical defaults

Start short. A profiler step is advanced once per env step in GeneLab's RL loop, so traces can grow
quickly on large vectorized runs.

## Distributed runs

Only the main process writes profiler traces. Keep `--prof-active` small when profiling distributed
training.

## See also

- [Run RL Experiments](../best-practices/rl-experiments.md)
- [CLI Reference](../reference/cli.md)
