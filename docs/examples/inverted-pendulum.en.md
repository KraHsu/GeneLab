# Inverted Pendulum

The inverted pendulum example is the recommended first runnable task. It is small enough for smoke
tests but exercises the full GeneLab train/play path.

## Tasks

| Task id | Description |
|---|---|
| `GeneLab-Inverted-Pendulum-v0` | Single pole balancing (rsl_rl PPO). |
| `GeneLab-Double-Inverted-Pendulum-v0` | Two-link pole balancing (rsl_rl PPO). |
| `GeneLab-Inverted-Pendulum-Skrl-v0` | Single pole balancing — same env, **skrl** PPO backend. |
| `GeneLab-Inverted-Pendulum-Memory-Rnn-v0` | "Recall the target" — an **LSTM** policy on a task that needs memory. |
| `GeneLab-Inverted-Pendulum-Memory-Mlp-v0` | Same task, plain **MLP** — the baseline that structurally can't remember. |

## Installing and listing

```bash
uv pip install -e examples/inverted_pendulum
genelab list tasks
```

Without installation:

```bash
PYTHONPATH=examples/inverted_pendulum/src \
  genelab --import genelab_inverted_pendulum.tasks list tasks
```

## Running

```bash
genelab play GeneLab-Inverted-Pendulum-v0 --steps 64
genelab play GeneLab-Inverted-Pendulum-v0 --vis --steps 500
genelab train GeneLab-Inverted-Pendulum-v0 --num_envs 64 --max_iterations 2

# Same env, skrl PPO backend (selected purely by the agent cfg type):
genelab train GeneLab-Inverted-Pendulum-Skrl-v0 --num_envs 64 --max_iterations 4800
# skrl names checkpoints agent_<timesteps>.pt under the run's checkpoints/ dir:
genelab eval GeneLab-Inverted-Pendulum-Skrl-v0 logs/skrl/inverted_pendulum_skrl/<run>/checkpoints/agent_<N>.pt
```

> `genelab train` for the skrl task needs the optional `skrl` dependency
> installed; registering and listing the task does not.

## Recurrent (RNN) memory showcase

`GeneLab-Inverted-Pendulum-Memory-Rnn-v0` shows **when recurrence actually helps**. It is a
"recall the target" task: each episode a random cart target is **flashed in the observation
for only the first 5 steps**, then removed. The flash is too brief to drive the cart there
while it is visible, so the policy must *remember* the target and move to it afterwards. A
memoryless MLP, once the cue is gone, has no way to recover the target and falls back to the
centre; an LSTM stores it in its hidden state and drives the cart straight to it. (Velocities
stay observed, so the only thing needing memory is the target.)

This is the task category where recurrence is decisive — unlike plain *balancing*, which a
feedforward MLP solves even with velocities or loads hidden, because feedback stabilisation is
robust to unobserved state. Memory, not partial observability alone, is what an RNN buys you.

Train both and compare the post-cue tracking error (mean distance from the hidden target):

```bash
genelab train GeneLab-Inverted-Pendulum-Memory-Rnn-v0 --num_envs 2048 --max_iterations 400
genelab train GeneLab-Inverted-Pendulum-Memory-Mlp-v0 --num_envs 2048 --max_iterations 400
```

| Policy | Task | Post-cue tracking error (mean \|cart − target\|) |
|---|---|---|
| **LSTM (recurrent)** | `...-Memory-Rnn-v0` | **≈ 0.013** — reaches and holds the hidden target |
| MLP (baseline) | `...-Memory-Mlp-v0` | ≈ 0.78 — can't recall it, sits near centre |

(Targets are uniform in `[-0.8, 0.8]`; a policy that ignores the target scores ≈ its spread.)

!!! note "Why it trains (and earlier memory attempts didn't)"

    Recurrent PPO uses truncated BPTT of length `num_steps_per_env`. The memory cfg sets it to
    `100` — one whole ~100-step episode — so each BPTT window spans the full cue→reach
    dependency and the gradient can flow from the recall back to where the target was encoded.
    With the default 24-step window the LSTM never learns to associate the two.

## Code entry points

| File | Role |
|---|---|
| `tasks.py` | Registers robots, envs, and tasks. |
| `single/env_cfg.py` | Single-pole manager-based env config. |
| `single/memory_env_cfg.py` | The "recall the target" env (masked target cue + tracking reward). |
| `single/rnn_cfg.py` | LSTM + MLP PPO configs for the memory task. |
| `double/env_cfg.py` | Double-pole manager-based env config. |
| `mdp.py` | Example-specific reward / termination / memory-task terms. |

## See also

- [Tutorial](../tutorial.md)
- [Task design](../best-practices/task-design.md)
