# Inverted Pendulum

`examples/inverted_pendulum/` ships two PPO tasks for the classical cart-pole problem on flat
ground. The extension mirrors `examples/unitree/`: `ManagerBasedRlEnv` over Genesis, rsl_rl PPO,
a `BodyVelocitySensor` on the pole link, and the unified `genelab train` / `genelab play` CLI.

## Tasks

| Task id | Problem |
|---------|---------|
| `GeneLab-Inverted-Pendulum-v0` | Single inverted pole on a cart. |
| `GeneLab-Double-Inverted-Pendulum-v0` | Two stacked inverted poles on a cart. |

## Installation

Pick the `torch-*` extra that matches the hardware.

```bash
uv sync --extra torch-cu128
uv pip install -e examples/inverted_pendulum

uv run genelab list tasks
# -> GeneLab-Inverted-Pendulum-v0
# -> GeneLab-Double-Inverted-Pendulum-v0
```

## Single inverted pendulum

```bash
uv run genelab train GeneLab-Inverted-Pendulum-v0 \
    --num-envs 4096 --max-iterations 150

uv run genelab play  GeneLab-Inverted-Pendulum-v0 \
    --checkpoint logs/rsl_rl/inverted_pendulum_flat/<run>/model_150.pt --vis
```

`--checkpoint` makes `play` route through the RL runner with `--agent trained` by default.

## Double inverted pendulum

```bash
uv run genelab train GeneLab-Double-Inverted-Pendulum-v0 \
    --num-envs 4096 --max-iterations 300

uv run genelab play  GeneLab-Double-Inverted-Pendulum-v0 \
    --checkpoint logs/rsl_rl/double_inverted_pendulum_flat/<run>/model_300.pt --vis
```

## Sensor and underactuation

Only the cart slide joint is PD-controlled. The pole hinges default to `kp=0, kv=0` so the
pendulum stays underactuated and the policy must learn balance through cart motion alone. A
`BodyVelocitySensor` attached to the top pole supplies a noisy angular-velocity observation
(corrupted with `Unoise` in the policy group, clean in the critic group).

## Interactive disturbance

Play mode launches a single environment (`num_envs=1`) and enables Genesis'
`MouseInteractionPlugin`. Left-click on the cart or pole and drag — a spring force pulls the
clicked link toward the cursor while the policy keeps balancing. Scroll wheel rotates the drag
plane around the surface normal. Release the button to remove the force.

!!! tip "Smoke-test budget"
    A 5–10 iteration run with `--num-envs 64 --max-iterations 5` is enough to validate wiring
    end-to-end. The reward signal will still be noisy at that scale; convergence requires the
    150 / 300 iteration budgets above.

## Logs

Both tasks write to `logs/rsl_rl/<experiment>/<timestamp>_/` like the Unitree examples:

- `params/env.json` and `params/agent.json` — frozen configs at run time.
- `model_<iter>.pt` — checkpoints saved every `save_interval` iterations.
- TensorBoard event files alongside the checkpoints.

## See also

- [Unitree G1 quickstart](../getting-started/quickstart.md#unitree-g1)
- [Sensors](../concepts/sensors.md)
- [Play and Train CLI](../cli/play-train.md)
