# Inverted pendulum examples

GeneLab extension that ships two PPO tasks for the classical cart-pole problem on flat ground:

- **`GeneLab-Inverted-Pendulum-v0`** — balance a single inverted pole on a cart.
- **`GeneLab-Double-Inverted-Pendulum-v0`** — stabilise two stacked inverted poles on a cart.

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

## Notes

- Only the cart slide joint is PD-controlled. The pole hinges default to `kp=0, kv=0` so the
  pendulum stays underactuated and the policy must learn balance via cart motion alone.
- The observation group corrupts joint position, joint velocity, and pole angular velocity with
  `Unoise`. The critic group sees the same features without corruption.
- Logs land under `logs/rsl_rl/<experiment>/<timestamp>_/` exactly like the Unitree examples,
  with `params/env.json`, `params/agent.json`, and `model_<iter>.pt` files.
