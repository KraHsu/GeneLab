# Wuji Hand Example

This directory is a normal external GeneLab project for the Wuji dexterous hand. It registers:

- `GeneLab-Wuji-Hand-Playback-v0` — fixed-trajectory playback of the Wuji hand.
- `Genelab-Reorient-Wuji-Hand-v0` — SO(3) in-hand cube reorientation, trainable with RSL-RL PPO (see [docs](../../docs/examples/wuji-reorient.en.md)).

From the repository root, run the example without installing this package by importing its
registration module:

```bash
PYTHONPATH=examples/wuji/src uv run genelab --import genelab_wuji.tasks list tasks
PYTHONPATH=examples/wuji/src uv run genelab --import genelab_wuji.tasks play GeneLab-Wuji-Hand-Playback-v0 --steps 240
```

Or install the example project once, then use the entry-point auto-loading path:

```bash
uv pip install -e examples/wuji
uv run genelab list tasks
uv run genelab play GeneLab-Wuji-Hand-Playback-v0 --steps 240
```

Use `--no-entry-points` when you want to verify core GeneLab without installed extensions.

## Deploy (sim2real)

A Genesis-native sim2real toolchain takes a trained reorient policy to the real
(or mock) Wuji hand — real2sim cube tracking, a Hikvision cube observer, calibration,
and the ONNX deploy control loop. It installs a unified **`wuji`** console entry:

```bash
uv pip install -e 'examples/wuji[deploy]'          # + [deploy-vision] / [deploy-hand] as needed
uv run wuji --help                                  # list commands
uv run wuji play --ckpt policy.onnx --real --goal-mode random
```

| command | purpose |
|---|---|
| `wuji check` / `home` | read-only hand-bridge test / ramp to grasp pose |
| `wuji observer` | Hikvision camera → ArUco cube pose → ZMQ |
| `wuji viewer` | real2sim Genesis mirror of the observed cube |
| `wuji calib` | calibration viewer (live hand + cube vs. digital twin) |
| `wuji play` | deploy control loop (real/mock) + goal modes + success monitor |

See **[`src/genelab_wuji/deploy/README.md`](src/genelab_wuji/deploy/README.md)** for the
full pipeline, ZMQ wiring, install extras, and the Hikvision MVS SDK setup.
