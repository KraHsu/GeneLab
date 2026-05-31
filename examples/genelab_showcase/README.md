# GeneLab visual showcase

Play-only GeneLab extension that wires every M1–M4 building block into an eight-task viewer
demo. Each task drops a real robot (Franka or Unitree G1) into a `ManagerBasedRlEnv`,
runs a scripted action loop, and dumps the relevant sensor / curriculum / terrain state
so the visual behaviour can be checked by eye.

- `GeneLab-Sensors-Showcase-v0` — Franka with `CameraSensor` (RGB+depth → PNG), `IMUSensor`, `FrameTransformerSensor`, and `ForceTorqueSensor`.
- `GeneLab-RayCast-Showcase-v0` — Franka with three `RayCastSensor` instances mounted side by side (`GridPattern` + `RingPattern` + `HemispherePattern`), all three dumped in parallel.
- `GeneLab-Contact-Showcase-v0` — Unitree G1 with `ContactSensor`, air-time tracking on both feet.
- `GeneLab-Terrain-Showcase-v0` — Unitree G1 dropped on a 1×5 row that tiles every built-in sub-terrain (`flat` + `stairs` + `rough` + `slope` + `wave`) in a single scene.
- `GeneLab-Curriculum-Showcase-v0` — Unitree G1 on a 5×5 RandomRough grid, scripted XY drift drives `terrain_levels_vel` promotion.
- `GeneLab-Actuator-Showcase-v0` — Franka with `IdealPDActuator` on the arm (force-channel control), prints joint-tracking error against a sine target.
- `GeneLab-MlpResidual-Actuator-Showcase-v0` — Franka with `MlpResidualActuator` on the arm (DC-motor base + TorchScript residual network).
- `GeneLab-Recording-Showcase-v0` — Franka with live PyQt / Matplotlib plots and NPZ / CSV data dumps from an IMU.

## Installation

```bash
uv pip install -e examples/genelab_showcase
uv run genelab list tasks | grep Showcase
```

## Running a showcase

The shared invocation pattern is `genelab play <task> --vis --steps <N>`. Each task
falls through to a custom `play()` that builds the env once, runs the scripted action
loop, and writes a `logs/showcase/<task>/` directory.

```bash
# RGB-D dump (requires Linux x86-64 + CUDA + BatchRenderer wiring).
uv run genelab play GeneLab-Sensors-Showcase-v0 --vis --steps 200

# Ray-cast: all three patterns render and dump in parallel; the log carries
# one block per sensor.
uv run genelab play GeneLab-RayCast-Showcase-v0 --vis --steps 200

# Terrain: the 1×5 row hits every sub-terrain in one scene.
uv run genelab play GeneLab-Terrain-Showcase-v0 --vis --steps 200
```

Each task's env cfg lives under `src/genelab_showcase/<feature>/env_cfg.py` —
swap a `RayCastSensorCfg.pattern`, retune a `TerrainGeneratorCfg.sub_terrains`
entry, or change a `RewardTermCfg` weight there. The CLI exposes the standard
override grammar (`--vis`, `--steps`, `--env.simulation.num_envs`, etc.) for
the simulation-level knobs; deeper structural changes belong in the cfg
source.

See `docs/examples/showcase.{en,zh}.md` for the full expected-output catalogue.

## Layout

```
examples/genelab_showcase/
├── pyproject.toml
├── README.md
└── src/genelab_showcase/
    ├── tasks.py               # register() + RegisteredTask dispatcher
    ├── runner.py              # ShowcaseRunner base class
    ├── sensors/               # Camera + IMU + FrameTransformer + ForceTorque
    ├── raycast/               # Grid / Ring / Hemisphere
    ├── contact/               # G1 foot air-time
    ├── terrain/               # 5 sub-terrains + mixed
    ├── curriculum/            # terrain_levels_vel
    ├── actuators/             # IdealPD (force control) + MlpResidual (DC-motor + residual)
    └── recording/             # live PyQt/MPL plots + NPZ/CSV IMU dump
```

The runtime entry point is `genelab_showcase.tasks:register`, auto-discovered by
GeneLab when the package is installed.
