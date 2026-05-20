# Reference Runs

This page is the **reproducibility ground truth** for GeneLab's bundled tasks.
It lists, per registered task and per seed, the converged return, the
convergence step count, and the wall-clock budget — the numbers you should
expect to land on when you `clone → train → eval` against the same
configuration.

> **Status — 2026-05-20**: the **reproduction protocol is final**; the
> **reference numbers are TBD** and tracked under ROADMAP M1.7. Treat the
> tables below as a schema; the populated PR will follow once the runs are
> done on a stable Genesis pin. If you have run these yourself, attach the
> numbers and curves to the GitHub issue tracking M1.7.

## Reproduction protocol

Every reference run uses GeneLab's `--seeds` fan-out and the deterministic
`genelab eval` for the final numbers:

```bash
# 1. Train three seeds in parallel (one Python process per seed; parallel=3
#    will saturate one machine if num_envs is moderate).
genelab train <TASK> \
    --num_envs <N> --max_iterations <ITERS> \
    --seeds 1,2,3 --parallel 3 \
    --log_dir logs/reference/<TASK>/<DATE>

# 2. Evaluate each seed's final checkpoint deterministically.
for s in 1 2 3; do
  genelab eval <TASK> \
    "logs/reference/<TASK>/<DATE>/seed_${s}/model_final.pt" \
    --num-envs 64 --episodes 100 --seed 0 \
    --out "logs/reference/<TASK>/<DATE>/seed_${s}/eval.json"
done
```

The `eval.json` files are the source of truth for the numbers in the tables.

Hardware: one CUDA GPU (≥ 12 GB VRAM) for training. CPU-only eval works for
the deterministic rollout step but is much slower than GPU-vectorized.

## Reference tasks

The five tasks tracked here cover GeneLab's bundled locomotion +
manipulation lines:

| Task ID | Backend (default agent) | Notes |
|---|---|---|
| `GeneLab-Inverted-Pendulum-v0` | rsl_rl PPO | Tiny cartpole; ~5 min on 64 envs. Useful as a smoke target. |
| `GeneLab-Double-Inverted-Pendulum-v0` | rsl_rl PPO | Harder cartpole; ~15 min on 64 envs. |
| `GeneLab-G1-Velocity-v0` | rsl_rl PPO | Unitree G1 velocity tracking; ~2 h on 4096 envs. |
| `GeneLab-Franka-Pick-And-Place-v0` | rsl_rl PPO (joint-space) | Manipulation joint-space variant; ~1 h. |
| `GeneLab-Franka-Pick-And-Place-Cartesian-v0` | rsl_rl PPO (Cartesian IK) | Same task via DifferentialIK; converges faster. |

## Reference numbers

### `GeneLab-Inverted-Pendulum-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |

### `GeneLab-Double-Inverted-Pendulum-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |

### `GeneLab-G1-Velocity-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |

### `GeneLab-Franka-Pick-And-Place-v0`

| Seed | Final `return_mean` | `return_std` | `success_rate` | Convergence iter | Train wall-clock |
|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |

### `GeneLab-Franka-Pick-And-Place-Cartesian-v0`

| Seed | Final `return_mean` | `return_std` | `success_rate` | Convergence iter | Train wall-clock |
|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |

## Training curves

Curves are exported from TensorBoard once the reference runs land. The
expected location:

```
logs/reference/<TASK>/<DATE>/seed_<S>/
├── events.out.tfevents.*   # TensorBoard
├── ckpts/                  # checkpoints (or `model_<N>.pt` under the dir,
│                           #   depending on backend)
├── eval.json               # written by `genelab eval`
└── (optional) curves.png   # screenshot used in this doc
```

Until the runs are done, this section is intentionally empty — the schema
above is what populated PRs should match.

## Methodology notes

- **Seeds 1, 2, 3 are GeneLab's canonical triplet.** Any task in this doc
  that ships with different seeds should explain why (e.g. seed 0 hit a
  degenerate Genesis init on this task).
- **Eval seed is fixed at 0.** This ensures the deterministic eval rollout
  is the same trajectory across seeds and across re-runs of this protocol —
  the variance in `return_mean` then reflects training variance only.
- **No `success_rate` for locomotion** at this revision. Locomotion tasks
  ship without `extras["is_success"]`; the doc reports `null` rather than
  inventing a threshold. Manipulation tasks (Franka) emit `is_success`
  from the goal-reach termination, so the field is populated there.
- **Genesis version pin.** The version used to produce these numbers is
  recorded at the top of each `eval.json` (via `evaluated_at` and the
  `params/env.json` snapshot in the same directory). Re-running with a
  different Genesis is *not* expected to reproduce the numbers exactly.

## What this doc is not

- It is **not** a benchmark suite — that is M3.8 (`genelab benchmark`).
- It is **not** a leaderboard. The numbers here are GeneLab's own
  reproducibility check; community submissions go to the benchmark suite
  once it lands.
- It is **not** a tuning guide. See `best-practices/rl-experiments` for
  curriculum, DR, and reward weight choices that are upstream of these
  numbers.
