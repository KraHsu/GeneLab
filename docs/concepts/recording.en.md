# Recording and live plotting

`genelab.recording` packages Genesis's recorder + live-plotter stack behind a
declarative `RecordingCfg`. A scene can declare one or more recordings, each pairing a
data source (sensor, articulation field, or custom callable) with one or more output
sinks (PyQt/Matplotlib plot, NPZ/CSV file, MP4/AVI video). The Genesis recorder pipeline
does the actual sampling on its own threads — Genelab just describes what to record and
wires the env lifecycle.

## Quick start

```python
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.recording import NPZFileCfg, PyQtPlotCfg, RecordingCfg
from genelab.sensor import IMUSensorCfg

scene = InteractiveSceneCfg(
    sensors=(IMUSensorCfg(name="cart_imu", link_name="cart"),),
    recordings=(
        RecordingCfg(
            name="cart_acc",
            source="cart_imu",
            field="lin_acc_b",
            outputs=(
                PyQtPlotCfg(title="cart linear acc", labels=("ax", "ay", "az")),
                NPZFileCfg(filename="logs/cart_acc.npz"),
            ),
        ),
    ),
)
```

Running `env.step(...)` then samples `cart_imu.data.lin_acc_b` once per control step,
streams it to a live PyQt window, and accumulates an `.npz` file written when the env
is closed. No manual data plumbing in your runner.

## Mental model

* **One `RecordingCfg` = one source × N sinks.** Outputs share the same data callable.
* **Registration is scene-build-time.** Genesis's `add_recorder` is `@assert_unbuilt`,
  so recordings land in `InteractiveSceneCfg` (next to `sensors`), not on the env-level
  `*_cfg` dicts. A small `RecorderBridge` carries the not-yet-bound env reference
  through to the callables.
* **Sampling happens inside `gs_scene.step()`.** Recorders run on Genesis threads —
  PyQt plots update without blocking the env loop.

## Sources

A `RecordingCfg.source` is one of:

* **Sensor name (string).** Combined with `field=...` to pick a tensor from the sensor's
  `data` payload (`"lin_acc_b"`, `"orientation"`, `"rgb"`, etc.). If `field` is omitted,
  `sensor.data` must already be a plain tensor — dataclass payloads raise a clear error.
* **Callable.** Takes either zero arguments or a single `env` argument; arity is
  detected once at registration. Useful for raw articulation state or computed signals
  that don't have a sensor wrapper.

```python
# sensor + dotted field
RecordingCfg(name="imu_ori", source="imu", field="orientation", outputs=(...,))

# nullary callable
RecordingCfg(name="wall_clock", source=time.monotonic, outputs=(...,))

# env-aware callable
RecordingCfg(
    name="q0",
    source=lambda env: env.robot_state.joint_pos[:, 0],
    outputs=(...,),
)
```

`RecordingCfg.env_idx` (default 0) squeezes a multi-env leading dimension to a single
env before each sample reaches the sinks; pass `env_idx=None` to keep the full batch
(useful when dumping all envs to NPZ).

## Output reference

| Cfg | Backend | When to use |
|-----|---------|-------------|
| `PyQtPlotCfg` | `pyqtgraph` (threaded) | Default for live plots — works on every platform Genesis supports. |
| `MPLPlotCfg` | `matplotlib` | Fallback when PyQt is unavailable; runs on the main thread on macOS (may stutter). |
| `NPZFileCfg` | `numpy.savez_compressed` | Buffered; one file per recording, flushed on `env.close()` (or per-episode with `save_on_reset=True`). |
| `CSVFileCfg` | `csv.writer` | Row-streamed; pass `header=("a","b",...)` for column names, `save_every_write=True` to flush each row. |
| `VideoFileCfg` | `cv2.VideoWriter` | MP4/AVI from a `CameraSensor`; only pairs with camera sources. |

All five cfgs accept the common `hz`, `buffer_size`, and `buffer_full_wait_time` knobs
from `RecorderOptions`. Setting `hz` explicitly always wins over the Genelab defaults.

## Sampling cadence and the sensor cache

`Sensor.data` is cached per control step. If a recorder sampled every physics tick it
would record `decimation` identical rows per control step. Genelab fixes this for you:

* **Sensor-name sources** default to `hz = 1 / (sim.dt * decimation)` (the control rate).
* **Callable sources** default to `hz = None` (every physics tick) — the assumption is
  that the callable reads raw articulation state that genuinely changes each tick.

Override either by setting `hz` on the output cfg directly:

```python
NPZFileCfg(filename="raw.npz", hz=200.0)  # explicit 200 Hz regardless of sim.dt
```

## Episode boundaries and `save_on_reset`

When at least one output has `save_on_reset=True`, the env's auto-reset path calls
`gs_scene._recorder_manager.reset()` to flush and rotate file outputs (NPZ counter,
new CSV header row). The very first reset inside `ManagerBasedRlEnv.__init__` is
skipped so file counters start at 0, matching user expectations
(`pole_0.npz`, `pole_1.npz`, …). The final partial buffer is flushed by
`env.close()`.

Plotters are unaffected by reset (history is rolling).

## Multi-env

Plotters always read a single env. Use `env_idx` to pick which one — the default 0 is
almost always what you want when scaling envs for RL while debugging.

File writers can record all envs at once: set `env_idx=None` and they store the full
`(num_envs, ...)` tensor. Plot outputs combined with `env_idx=None` will fail at the
first sample — split the recording.

## Cameras and video

A `CameraSensor` source paired with a single `VideoFileCfg` writes a one-env video:

```python
RecordingCfg(
    name="wrist_video",
    source="wrist_cam",
    outputs=(VideoFileCfg(filename="logs/wrist.mp4", env_idx=0, fps=30),),
)
```

Caveats:

* The camera must have `render_rgb=True` — depth-only cameras are rejected at runtime.
* `BatchRenderer` is Linux x86-64 + CUDA only. On macOS the default Rasterizer still
  produces a single-env `(1, H, W, 3)` tensor that `env_idx=0` slices correctly.
* Pairing a plotter with a camera source is rejected at registration — plotters can't
  consume H×W×3 frames.

## Custom callables and arbitrary signals

Any quantity Genelab exposes on the env can become a recording. Examples:

```python
# joint torque (post-actuator)
RecordingCfg(
    name="joint_effort",
    source=lambda env: env.robot_state.dof_force[:, env.joint_names.index("joint1")],
    outputs=(NPZFileCfg(filename="logs/effort.npz"),),
)

# reward term snapshot
RecordingCfg(
    name="track_reward",
    source=lambda env: env.reward_manager._term_sums["track_lin_vel"],
    outputs=(PyQtPlotCfg(title="track reward"),),
)
```

The callable runs on every recorder tick (every physics step by default). Keep it
cheap — heavy compute belongs in a Genelab sensor.

## Optional dependencies

Live plot backends live behind a `recording` extra on Genelab itself:

```bash
pip install "genelab[recording]"   # pulls pyqtgraph + PyQt5 + matplotlib
```

The showcase package depends on `genelab[recording]` transitively, so
`uv pip install -e examples/genelab_showcase` already pulls them in — no manual step.
File-only recordings (NPZ / CSV / VideoFile) need no extra dependencies beyond
Genelab's existing stack (numpy + PyAV, both Genesis transitive deps).

## See also

- [Sensors](sensors.md)
- [Scene and entities](scene.md)
- The `Showcase-Recording` task in [Examples → Showcase](../examples/showcase.md).
