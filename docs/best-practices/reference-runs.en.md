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

## Reference tasks

The five tasks tracked here cover GeneLab's bundled locomotion +
manipulation lines:

| Task ID | Backend (default agent) | Budget | Notes |
|---|---|---|---|
| `GeneLab-Inverted-Pendulum-v0` | rsl_rl PPO | 150 iter × 4096 envs | Tiny cartpole; sanity smoke target. |
| `GeneLab-Double-Inverted-Pendulum-v0` | rsl_rl PPO | 300 iter × 4096 envs | Harder cartpole. |
| `Genelab-Velocity-Flat-Unitree-G1-v0` | rsl_rl PPO | 30k iter × 4096 envs | Unitree G1 velocity tracking on flat ground. |
| `Genelab-Tracking-Flat-Unitree-G1-v0` | rsl_rl PPO | 30k iter × 4096 envs | Unitree G1 motion-tracking on flat ground. |
| `GeneLab-Franka-Pick-And-Place-v0` | sb3 SAC + HER | 2M timesteps × 2048 envs | Goal-conditioned manipulation; needs offline demo prefill (see protocol below). |

> **Note**: dev pruned the example to a single SAC+HER setup. The earlier
> `Cartesian-v0` / `skrl-v0` / `sb3-v0` / `sb3-her-v0` Franka variants no
> longer register and are intentionally not part of this set.

## Reproduction protocol

### Common path (4 of 5 tasks)

Cartpole + G1 tasks are rsl_rl PPO; their reference runs use the multi-seed
CLI directly:

```bash
# 1. Train three seeds (parallel=3 only for cartpole-sized tasks; G1 needs
#    parallel=1 on a single GPU to avoid OOM).
genelab train <TASK> \
    --seeds 1,2,3 --parallel <P> \
    --log_dir logs/reference/<TASK>/<DATE>

# 2. Deterministic eval against each seed's final checkpoint.
for s in 1 2 3; do
  genelab eval <TASK> \
    "logs/reference/<TASK>/<DATE>/seed_${s}/model_final.pt" \
    --num-envs 64 --episodes 100 --seed 0 \
    --out "logs/reference/<TASK>/<DATE>/seed_${s}/eval.json"
done
```

`eval.json` files are the source of truth for the table numbers.

### Franka SAC+HER path

`GeneLab-Franka-Pick-And-Place-v0` is goal-conditioned SAC+HER and needs an
offline demo prefill before training, otherwise the cold-start replay
buffer never sees a successful trajectory:

```bash
# 1. Collect demos via the scripted FSM (one-shot, seed-independent).
uv run python -m genelab_franka_pick_and_place.collect_demos \
    --num-envs 32 --steps 6400 \
    --out logs/reference/franka-pp/demos.npz

# 2. Train three seeds — each child reads the demo file via
#    GENELAB_SB3_DEMO_PATH (or set agent.demo_path in cfg).
GENELAB_SB3_DEMO_PATH=logs/reference/franka-pp/demos.npz \
  genelab train GeneLab-Franka-Pick-And-Place-v0 \
    --seeds 1,2,3 --parallel 1 \
    --log_dir logs/reference/franka-pp/<DATE>

# 3. Eval each seed's saved model.zip (SB3's native format).
for s in 1 2 3; do
  genelab eval GeneLab-Franka-Pick-And-Place-v0 \
    "logs/reference/franka-pp/<DATE>/seed_${s}/model.zip" \
    --num-envs 64 --episodes 100 --seed 0 \
    --out "logs/reference/franka-pp/<DATE>/seed_${s}/eval.json"
done
```

The Franka task **cannot** currently be exported via `genelab export` —
HER's `Dict` observation is on the M1.3 limitations list. Removing that
limitation is tracked as a future M-series follow-up.

### Hardware

One CUDA GPU (≥ 12 GB VRAM) for training. CPU-only eval works for the
deterministic rollout step but is much slower than GPU-vectorized.

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

### `Genelab-Velocity-Flat-Unitree-G1-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |

### `Genelab-Tracking-Flat-Unitree-G1-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |

### `GeneLab-Franka-Pick-And-Place-v0` (SAC+HER, demo-prefilled)

| Seed | Final `return_mean` | `return_std` | `success_rate` | Convergence timestep | Train wall-clock |
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
