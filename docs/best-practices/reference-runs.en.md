# Reference Runs

This page is the **reproducibility ground truth** for GeneLab's bundled tasks.
It lists, per registered task and per seed, the converged return, the
convergence step count, and the wall-clock budget — the numbers you should
expect to land on when you `clone → train → eval` against the same
configuration.

## Reference tasks

The five tasks tracked here cover GeneLab's bundled locomotion +
manipulation lines:

| Task ID | Backend (default agent) | Budget | Notes |
|---|---|---|---|
| `GeneLab-Inverted-Pendulum-v0` | rsl_rl PPO | 150 iter × 4096 envs | Tiny cartpole; sanity smoke target. |
| `GeneLab-Double-Inverted-Pendulum-v0` | rsl_rl PPO | 300 iter × 4096 envs | Harder cartpole. |
| `Genelab-Velocity-Flat-Unitree-G1-v0` | rsl_rl PPO | 30k iter × 4096 envs | Unitree G1 velocity tracking on flat ground. |
| `Genelab-Tracking-Flat-Unitree-G1-v0` | rsl_rl PPO | 30k iter × 4096 envs | Unitree G1 motion-tracking on flat ground. |
| `GeneLab-Franka-Pick-And-Place-v0` | sb3 SAC + HER | 2M timesteps × 64 envs | Goal-conditioned manipulation; needs offline demo prefill (see protocol below). |

## Reproduction protocol

### Common path (4 of 5 tasks)

Cartpole + G1 tasks are rsl_rl PPO; their reference runs use the multi-seed
CLI:

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
#    --num-envs must match the task's train num_envs (currently 64); the
#    prefill loader asserts the shapes match.
uv run python -m genelab_franka_pick_and_place.collect_demos \
    --num-envs 64 --steps 1000 \
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

The Franka task **cannot** currently be exported via `genelab export`.
Export supports flat-tensor observations only, while SAC+HER uses a
goal-conditioned `Dict` observation.

### Hardware

One CUDA GPU (≥ 12 GB VRAM) for training. CPU-only eval works for the
deterministic rollout step but is much slower than GPU-vectorized.

## Reference numbers

### `GeneLab-Inverted-Pendulum-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 39.944 | 0.026 | 150 | ~21 min | 10.3 s |
| 2 | 39.978 | 0.002 | 150 | ~20 min | 10.1 s |
| 3 | 39.991 | 0.001 | 150 | ~19 min | 10.1 s |

Eval `length_mean = 1000.0` for all seeds (episode hits the time-limit cap
without falling), so the policy is solved at the budget cap. `success_rate`
is `null` (task does not publish `extras["is_success"]`).

### `GeneLab-Double-Inverted-Pendulum-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 59.980 | 0.007 | 300 | ~85 min | 12.2 s |
| 2 | 59.986 | 0.003 | 300 | ~88 min | 14.2 s |
| 3 | 59.987 | 0.002 | 300 | ~85 min | 12.6 s |

Eval `length_mean = 1200.0` for all seeds. `success_rate` is `null` (same
reason as IP).

### `Genelab-Velocity-Flat-Unitree-G1-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 112.419 | 4.647 | 30 000 | ~18.7 h | 143.0 s |
| 2 | 93.417 | 3.921 | 30 000 | ~20.6 h | 161.0 s |
| 3 | 92.028 | 4.162 | 30 000 | ~19.8 h | 156.9 s |

Eval `length_mean = 1000.0` for all seeds (play_env `episode_length_s =
20 s` × 50 Hz). `success_rate` is `null`.

### `Genelab-Tracking-Flat-Unitree-G1-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 137.800 | 0.005 | 30 000 | ~20.8 h | 212.8 s |
| 2 | 138.047 | 0.004 | 30 000 | ~20.6 h | 216.8 s |
| 3 | 138.122 | 0.007 | 30 000 | ~20.9 h | 216.0 s |

Eval `length_mean = 1500.0`. The tracking play_env normally sets
`episode_length_s = 1e9` for infinite viewer playback; `genelab eval`
clamps that to 30 s, so 30 s × 50 Hz = 1500 steps per episode, all hitting
the cap without termination. Very tight std across seeds — the converged
policy follows the motion clip on track under the 30 s window. `success_rate`
is `null`.

### `GeneLab-Franka-Pick-And-Place-v0` (SAC+HER, demo-prefilled)

| Seed | Final `return_mean` | `return_std` | `success_rate` | Convergence timestep | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|---|
| 1 | −19.264 | 33.334 | 0.89 | 2 000 000 (budget cap) | ~68 min | 15.3 s |
| 2 |  −4.626 |  8.297 | 1.00 | 2 000 000 (budget cap) | ~63 min | 14.3 s |
| 3 |  −4.102 |  7.644 | 1.00 | 2 000 000 (budget cap) | ~64 min | 18.7 s |

Eval `length_mean = 100.0` (fixed episode length). Mean
`success_rate ≈ 0.963 ± 0.052` across the three seeds; the two perfect
seeds reflect a fully solved policy, the 0.89 seed still misses ~11 %
of episodes from end-effector orientation drift on harder goal poses.

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

- It is **not** a benchmark suite or leaderboard. The numbers here are
  GeneLab's own reproducibility check.
- It is **not** a tuning guide. See `best-practices/rl-experiments` for
  curriculum, DR, and reward weight choices that are upstream of these
  numbers.
