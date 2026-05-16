# Showcase

`examples/genelab_showcase/` ships seven play-only tasks that exercise every M1–M4
building block in a single command. Each task drops a real robot — Franka or
Unitree G1 — into a minimal `ManagerBasedRlEnv`, runs a scripted action loop, and
writes a per-feature evidence file under `logs/showcase/<slug>/`. The intent is
human visual / numerical verification, not training.

## Tasks

| Task id | Robot | Feature exercised |
|---|---|---|
| `GeneLab-Sensors-Showcase-v0` | Franka | `CameraSensor` (RGB+depth PNG dump), `IMUSensor`, `FrameTransformerSensor` |
| `GeneLab-RayCast-Showcase-v0` | Franka | `RayCastSensor` × 3 patterns (`GridPattern`, `RingPattern`, `HemispherePattern`) |
| `GeneLab-Contact-Showcase-v0` | Unitree G1 | `ContactSensor` with `track_air_time=True` on both ankle-roll links |
| `GeneLab-Terrain-Showcase-v0` | Unitree G1 | `TerrainGeneratorCfg` 1×5 row of the five built-in sub-terrains |
| `GeneLab-Curriculum-Showcase-v0` | Unitree G1 | `terrain_levels_vel` curriculum on a 5×5 RandomRough grid |
| `GeneLab-Actuator-Showcase-v0` | Franka | `IdealPDActuator` arm (force-channel control); tracking-error dump |
| `GeneLab-Recording-Showcase-v0` | Franka | `genelab.recording` — live PyQt + MPL plots, NPZ + CSV dumps from an IMU |

## Installation

```bash
uv sync --extra torch-cu128       # or whichever torch-* extra matches the host
uv pip install -e examples/genelab_showcase

uv run genelab list tasks | grep Showcase
```

The showcase extension auto-loads through the `genelab.extensions` entry point once
installed; no `--import` flag is needed.

## Sensors

```bash
uv run genelab play GeneLab-Sensors-Showcase-v0 --vis --steps 200
```

Joint 1 is driven by a slow sine (±0.5 rad over a 4 s period) while the rest of the
arm holds the Menagerie home pose. Every 20 control steps the runner dumps:

- `logs/showcase/sensors/rgb_<step>.png` — wrist-camera RGB at 160×120.
- `logs/showcase/sensors/depth_<step>.png` — wrist-camera depth as 16-bit gray scaled
  to the camera's [near, far] = [0.02, 2.5] m clipping range.
- `logs/showcase/sensors/frame.log` — single-line record of IMU orientation, IMU
  linear / angular acceleration, and end-effector + link 7 pose in the base frame.

!!! warning "BatchRenderer is the only parallel-env backend"
    `InteractiveSceneCfg.batch_render=True` is already set in the env config. The
    sensor showcase therefore requires Linux x86-64 + CUDA + a Genesis build with
    Madrona compiled in. On other platforms the env constructor will raise as soon
    as Genesis allocates the renderer.

## Ray-cast patterns

```bash
uv run genelab play GeneLab-RayCast-Showcase-v0 --vis --steps 200
```

Three `RayCastSensor` instances co-exist on the Franka base — one `GridPattern`
(81 rays, 0.8×0.8 m), one `RingPattern` (32 horizontal × 1 vertical, full ±180°), and
one `HemispherePattern` (128 Fibonacci-distributed rays, 70° polar cap pointing down).
Every 20 steps the runner appends a three-line block to
`logs/showcase/raycast/distances.log`:

```
step=0020
  raycast_grid: rays=81 min=… mean=… max=…
  raycast_ring: rays=32 min=… mean=… max=…
  raycast_hemi: rays=128 min=… mean=… max=…
```

`max` clamps at the configured `max_distance`; rays that miss the ground plane (rare
when the pole axis points down) take that ceiling value.

## Contact and air time

```bash
uv run genelab play GeneLab-Contact-Showcase-v0 --vis --steps 200
```

The G1 settles into its default standing pose; both feet make ground contact almost
immediately. `logs/showcase/contact/air_time.log` records — every 20 steps — the
per-foot `found` flag, the running `current_contact_time` / `current_air_time`, and
the last completed `last_contact_time` / `last_air_time` snapshots.

A clean reset cycle every 4 s (`episode_length_s=4`) lets the air-time counters tick
through several boundaries so the snapshot semantics are visible in the log.

## Terrains

```bash
uv run genelab play GeneLab-Terrain-Showcase-v0 --vis --steps 200
```

A 1×5 sub-terrain row instantiates each built-in surface in a single scene:

| Column | Sub-terrain | Visible feature |
|---|---|---|
| 0 | `FlatPatchCfg` | Flat reference patch at z=0 |
| 1 | `PyramidStairsCfg(step_width=0.4, step_height=-0.08)` | Concentric descending stairs |
| 2 | `RandomRoughCfg(min=-0.08, max=0.08)` | Uniform bumps |
| 3 | `SlopeCfg(slope=-0.25)` | Linear incline |
| 4 | `WaveCfg(num_waves=2, amplitude=0.08)` | Sinusoidal swell |

G1 spawns at the row centre with default pose and is allowed to fall. A small
downward `GridPattern` ray-cast on the pelvis records the local heightfield response
into `logs/showcase/terrain/terrain.log`.

## Curriculum

```bash
uv run genelab play GeneLab-Curriculum-Showcase-v0 --vis --steps 400
```

16 G1 instances spawn on a 5×5 `RandomRoughCfg` grid where row index controls
roughness amplitude. The runner teleports half the envs 1.5 m forward of spawn every
30 control steps to trip the `walked > distance_threshold` branch on auto-reset; the
remaining envs sit still and demote toward level 0. `logs/showcase/curriculum/levels.log`
appends a per-env level vector plus the row histogram every 30 steps.

## Actuators

```bash
uv run genelab play GeneLab-Actuator-Showcase-v0 --vis --steps 200
```

Franka is rebuilt with `IdealPDActuator` driving the seven arm joints — same
`stiffness=400`, `damping=80` numbers as the asset-zoo default, but routed through
`control_dofs_force` instead of Genesis's implicit PD. A 0.8-rad sine target is sent
to joint 1; `logs/showcase/actuators/tracking.log` records target vs actual position
and joint velocity every 20 steps so the tracking quality is directly comparable to
the sensors showcase (which uses `ImplicitPDActuator`).

!!! tip "Smoke-test budget"
    Every showcase finishes in well under a minute at the default 200 / 400 steps on
    a single env. Increase `--steps` only when accumulating longer log evidence;
    none of the showcases benefit from large parallel-env counts.

## Logs

Each showcase writes to `logs/showcase/<slug>/` with the slugs `sensors`, `raycast`,
`contact`, `terrain`, `curriculum`, `actuators`. PNG dumps land alongside the text
logs so a single directory is all that needs archiving when comparing runs.

## See also

- [Sensors](../concepts/sensors.md)
- [Terrains](../concepts/terrains.md)
- [Scene and entities](../concepts/scene.md)
