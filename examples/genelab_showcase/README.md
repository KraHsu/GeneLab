# GeneLab visual showcase

Play-only GeneLab extension that wires every M1–M4 building block into a six-task viewer
demo. Each task drops a real robot (Franka or Unitree G1) into a `ManagerBasedRlEnv`,
runs a scripted action loop, and dumps the relevant sensor / curriculum / terrain state
so the visual behaviour can be checked by eye.

- `GeneLab-Sensors-Showcase-v0` — Franka with `CameraSensor` (RGB+depth → PNG), `IMUSensor`, `FrameTransformerSensor`.
- `GeneLab-RayCast-Showcase-v0` — Franka with `RayCastSensor`, pattern selectable via override (`grid` / `ring` / `hemisphere`).
- `GeneLab-Contact-Showcase-v0` — Unitree G1 with `ContactSensor`, air-time tracking on both feet.
- `GeneLab-Terrain-Showcase-v0` — Unitree G1 dropped on procedural terrain, sub-terrain selectable (`flat` / `stairs` / `rough` / `slope` / `wave` / `mixed`).
- `GeneLab-Curriculum-Showcase-v0` — Unitree G1 on a 5×5 RandomRough grid, scripted XY drift drives `terrain_levels_vel` promotion.
- `GeneLab-Actuator-Showcase-v0` — Franka run under `ImplicitPDActuator` and `IdealPDActuator` (overridable), prints joint-tracking error against a sine target.

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
# RGB-D dump (requires Linux x86-64 + CUDA + BatchRenderer wiring)
uv run genelab play GeneLab-Sensors-Showcase-v0 --vis --steps 200

# Ray-cast: swap pattern at the CLI
uv run genelab play GeneLab-RayCast-Showcase-v0 --vis --steps 200 \
    --env.scene.sensors.0.pattern_kind ring

# Terrain: swap sub-terrain at the CLI
uv run genelab play GeneLab-Terrain-Showcase-v0 --vis --steps 200 \
    --env.scene.terrain_kind stairs
```

See `docs/examples/showcase.{en,zh}.md` for the full expected-output catalogue.

## Layout

```
examples/genelab_showcase/
├── pyproject.toml
├── README.md
└── src/genelab_showcase/
    ├── tasks.py               # register() + RegisteredTask dispatcher
    ├── runner.py              # ShowcaseRunner base class
    ├── sensors/               # Camera + IMU + FrameTransformer
    ├── raycast/               # Grid / Ring / Hemisphere
    ├── contact/               # G1 foot air-time
    ├── terrain/               # 5 sub-terrains + mixed
    ├── curriculum/            # terrain_levels_vel
    └── actuators/             # ImplicitPD / IdealPD / DCMotor
```

The runtime entry point is `genelab_showcase.tasks:register`, auto-discovered by
GeneLab when the package is installed.
