# Profiling

GeneLab ships a thin wrapper around `torch.profiler` for measuring per-step cost on both
training and inference. The profiler is opt-in: `play` and `train` accept `--prof` flags,
and the matching `GENELAB_PROFILE` environment variables serve as a fallback. Traces are
written in TensorBoard format.

## Enabling the profiler

Both commands accept the same flags. A CLI flag wins over the env var when both are set.

```bash
uv run genelab train <task-id> --prof --prof-out logs/myprof
GENELAB_PROFILE=1 GENELAB_PROFILE_OUT=logs/myprof uv run genelab train <task-id>
```

## Flags

| Flag | Env-var equivalent | Effect |
|------|--------------------|--------|
| `--prof` | `GENELAB_PROFILE=1` | Turn the profiler on. |
| `--prof-out PATH` | `GENELAB_PROFILE_OUT` | Trace output directory (default `logs/torch_profile`). |
| `--prof-wait N` | `GENELAB_PROFILE_WAIT` | Steps skipped before each cycle (default `10`). |
| `--prof-warmup N` | `GENELAB_PROFILE_WARMUP` | Warmup steps inside the cycle (default `5`). |
| `--prof-active N` | `GENELAB_PROFILE_ACTIVE` | Steps actively recorded per cycle (default `10`). |
| `--prof-repeat N` | `GENELAB_PROFILE_REPEAT` | Cycle count (default `2`). |
| `--prof-record-shapes` | `GENELAB_PROFILE_RECORD_SHAPES=1` | Record tensor shapes for op-input attribution. |
| `--prof-with-stack` | `GENELAB_PROFILE_WITH_STACK=1` | Capture Python stack traces. High overhead. |

## Schedule defaults

A cycle covers `wait + warmup + active` steps, and `repeat` is the number of cycles. With
the defaults each cycle covers 25 steps and two cycles run before capture stops, so a
single `--prof` run records 20 active steps — enough to surface hot ops without producing
oversized traces.

For `train`, one step is one rollout call on `RslRlVecEnvWrapper.step`. For `play`, one
step is one policy/env iteration. To express the schedule in PPO iterations instead of
env steps, multiply `--prof-wait`, `--prof-warmup`, and `--prof-active` by
`num_steps_per_env`.

## Viewing traces

`genelab prof open` launches TensorBoard against a trace directory:

```bash
uv run genelab prof open logs/torch_profile --port 6006
```

The command refuses to start when the directory does not exist, and prints an install
hint when `tensorboard` is not on `PATH`.

## Distributed runs

!!! note "Rank-0 only"
    Under torchrun, only rank 0 writes traces. Other ranks see `maybe_profile()` as a
    no-op so workers do not overwrite each other's files. CLI flags propagate through the
    torchrun relaunch unchanged.

## See also

- [Play and Train](play-train.md)
- [Configs](../concepts/configs.md)
- [torch.profiler reference](https://pytorch.org/docs/stable/profiler.html)
