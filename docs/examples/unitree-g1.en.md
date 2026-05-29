# Unitree G1

The Unitree G1 example is the advanced robot path after the inverted pendulum tutorial. It shows how
GeneLab scales to humanoid locomotion and motion imitation.

## Tasks

| Task id | Shows |
|---|---|
| `Genelab-Velocity-Flat-Unitree-G1-v0` | Velocity tracking on flat ground. |
| `Genelab-Tracking-Flat-Unitree-G1-v0` | Reference motion tracking. |

## Installing

```bash
uv pip install -e examples/unitree
genelab list tasks
```

## Velocity tracking

```bash
genelab play Genelab-Velocity-Flat-Unitree-G1-v0 --vis --steps 500
genelab train Genelab-Velocity-Flat-Unitree-G1-v0 \
  --num_envs 4096 \
  --max_iterations 1500
```

Replay:

```bash
genelab play Genelab-Velocity-Flat-Unitree-G1-v0 \
  --checkpoint logs/rsl_rl/g1_velocity_flat/<run>/model_1500.pt
```

## Motion imitation

```bash
python -m genelab_unitree.replay_motion
genelab train Genelab-Tracking-Flat-Unitree-G1-v0 \
  --num_envs 4096 \
  --max_iterations 30000
```

The default clip is fetched through the asset zoo and cached locally.

## See also

- [RL experiments](../best-practices/rl-experiments.md)
- [Asset zoo](../concepts/asset_zoo.md)
