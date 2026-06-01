# Inverted pendulum examples

GeneLab extension that ships PPO tasks for the classical cart-pole problem on flat ground:

- **`GeneLab-Inverted-Pendulum-v0`** — balance a single inverted pole on a cart.
- **`GeneLab-Double-Inverted-Pendulum-v0`** — stabilise two stacked inverted poles on a cart.
- **`GeneLab-Inverted-Pendulum-Memory-Rnn-v0`** — "recall the target", a task that **needs
  memory**, solved by an LSTM. See [Recurrent showcase](#recurrent-rnn-memory-showcase).
- **`GeneLab-Inverted-Pendulum-Memory-Mlp-v0`** — the same task with a plain MLP, kept as the
  baseline that structurally cannot remember.

The training stack mirrors `examples/unitree/`: `ManagerBasedRlEnv` over Genesis, rsl_rl PPO,
`BodyVelocitySensor` for the noisy pole rate observation, and the unified `genelab train` /
`genelab play` CLI.

## Layout

```
examples/inverted_pendulum/
├── pyproject.toml
├── README.md
├── assets/                                # cart-pole MJCFs
│   ├── inverted_pendulum.xml
│   └── double_inverted_pendulum.xml
└── src/genelab_inverted_pendulum/
    ├── tasks.py                           # registers both tasks
    ├── mdp.py                             # cart-pole-specific reward / termination / event terms
    ├── single/                            # single-pendulum config (robot + env + PPO)
    └── double/                            # double-pendulum config (robot + env + PPO)
```

## Quickstart

```bash
# From the GeneLab repo root
uv sync --extra torch-cu128         # pick whichever torch flavor fits your GPU
uv pip install -e examples/inverted_pendulum

uv run genelab list tasks
# -> GeneLab-Inverted-Pendulum-v0
# -> GeneLab-Double-Inverted-Pendulum-v0
# -> GeneLab-Inverted-Pendulum-Memory-Rnn-v0
# -> GeneLab-Inverted-Pendulum-Memory-Mlp-v0
```

### Single inverted pendulum

```bash
uv run genelab train GeneLab-Inverted-Pendulum-v0 --num_envs 4096 --max_iterations 150
uv run genelab play  GeneLab-Inverted-Pendulum-v0 \
    --checkpoint logs/rsl_rl/inverted_pendulum_flat/<run>/model_150.pt
```

### Double inverted pendulum

```bash
uv run genelab train GeneLab-Double-Inverted-Pendulum-v0 --num_envs 4096 --max_iterations 300
uv run genelab play  GeneLab-Double-Inverted-Pendulum-v0 \
    --checkpoint logs/rsl_rl/double_inverted_pendulum_flat/<run>/model_300.pt
```

## Recurrent (RNN) memory showcase

`GeneLab-Inverted-Pendulum-Memory-{Rnn,Mlp}-v0` demonstrate **when recurrence actually helps**
— and, just as importantly, when it doesn't.

The task is "recall the target": each episode a random cart target is **flashed in the
observation for only the first 5 steps**, then removed. The flash is too brief to drive the
cart there while visible, so the policy must *remember* the target and move to it afterwards.
Both tasks share the same env and training budget; the only difference is the policy network:

- the **MLP** (`...-Memory-Mlp-v0`) has no memory — once the cue is gone it can't recover the
  target and drifts back to the centre;
- the **LSTM** (`...-Memory-Rnn-v0`) stores the target in its hidden state and drives straight
  to it.

```bash
uv run genelab train GeneLab-Inverted-Pendulum-Memory-Rnn-v0 --num_envs 2048 --max_iterations 400
uv run genelab train GeneLab-Inverted-Pendulum-Memory-Mlp-v0 --num_envs 2048 --max_iterations 400
```

Measured post-cue tracking error (mean distance from the hidden target; targets uniform in
`[-0.8, 0.8]`):

| Policy | Tracking error | Outcome |
|---|---|---|
| **LSTM (recurrent)** | **≈ 0.013** | reaches & holds the hidden target |
| MLP (baseline) | ≈ 0.78 | can't recall it, sits near centre |

This is the right kind of task to reach for an RNN. Plain **balancing** is *not* — a feedforward
MLP balances the cart-pole even with velocities (or a variable pole load) hidden, because
feedback stabilisation is robust to unobserved state. What an RNN buys you is **memory**, not
partial observability per se.

> **Why it trains.** Recurrent PPO uses truncated BPTT of length `num_steps_per_env`. The
> memory cfg sets it to `100` (≈ one full episode) so each BPTT window spans the whole cue→reach
> dependency — the gradient flows from the recall back to where the target was encoded. With the
> default 24-step window the LSTM never learns to associate the two.

## Notes

- Only the cart slide joint is PD-controlled. The pole hinges default to `kp=0, kv=0` so the
  pendulum stays underactuated and the policy must learn balance via cart motion alone.
- The observation group corrupts joint position, joint velocity, and pole angular velocity with
  `Unoise`. The critic group sees the same features without corruption.
- Logs land under `logs/rsl_rl/<experiment>/<timestamp>_/` exactly like the Unitree examples,
  with `params/env.json`, `params/agent.json`, and `model_<iter>.pt` files.
