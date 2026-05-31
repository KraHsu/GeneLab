"""Declarative configuration objects for recording sensor / actuator / articulation data.

Each :class:`RecordingCfg` describes one data source feeding one or more output sinks
(live plot, file writer, video). The recording pipeline is implemented on top of
Genesis's :class:`genesis.recorders.RecorderManager` — these dataclasses are translated
into Genesis ``RecorderOptions`` at scene-build time inside
:func:`genelab.recording.register.register_recorders`.

Plain Python dataclasses (no ``torch``, no ``genesis`` imports) so users can construct
recording configs at module-import time without booting either runtime.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class OutputCfg:
    """Base for every recorder sink. ``hz=None`` defers the rate decision to register-time.

    For sensor-name sources the registration code patches the resolved Genesis recorder
    to sample once per control step (``decimation`` physics ticks) — this is the only
    rate at which ``Sensor.data`` actually changes. For callable sources ``hz=None``
    samples every physics tick (the data func can read raw state on each one). Set
    ``hz`` explicitly to override either default.
    """

    hz: float | None = None
    buffer_size: int = 0
    buffer_full_wait_time: float = 0.1


@dataclass
class PyQtPlotCfg(OutputCfg):
    """Live PyQtGraph window. Thread-safe on every platform Genesis supports."""

    title: str = ""
    labels: tuple[str, ...] | dict[str, tuple[str, ...]] | None = None
    window_size: tuple[int, int] = (800, 600)
    history_length: int = 100


@dataclass
class MPLPlotCfg(OutputCfg):
    """Live Matplotlib window. Runs on the main thread on macOS (may stutter under the env loop)."""

    title: str = ""
    labels: tuple[str, ...] | dict[str, tuple[str, ...]] | None = None
    window_size: tuple[int, int] = (800, 600)
    history_length: int = 100


@dataclass
class MPLImagePlotCfg(OutputCfg):
    """Live Matplotlib image window (``imshow``), updated each sample via blit.

    Pairs with a :class:`~genelab.sensor.CameraSensor` source: set ``field="rgb"`` for the
    ``(H, W, 3)`` uint8 frame, or ``field="depth"`` for the ``(H, W)`` depth map (rendered
    as a heatmap). Unlike :class:`PyQtPlotCfg` / :class:`MPLPlotCfg` (1-D time series), this
    shows the 2-D frame itself. Main-thread only with an interactive backend; Genesis
    auto-detects the display, so it no-ops on a headless host.
    """

    title: str = ""
    window_size: tuple[int, int] = (480, 360)


@dataclass
class NPZFileCfg(OutputCfg):
    """Buffers samples and writes a compressed ``.npz`` at cleanup."""

    filename: str = ""
    save_on_reset: bool = False


@dataclass
class CSVFileCfg(OutputCfg):
    """Streams rows to a ``.csv`` file."""

    filename: str = ""
    header: tuple[str, ...] | None = None
    save_every_write: bool = False
    save_on_reset: bool = False


@dataclass
class VideoFileCfg(OutputCfg):
    """Writes camera frames to a single-env ``.mp4`` via Genesis's PyAV writer.

    Only pairs with :class:`~genelab.sensor.CameraSensor` sources. ``env_idx`` picks the
    env whose frames are written (default: 0). ``fps=None`` derives the output rate from
    the recorder's effective sampling rate. ``codec=""`` (the default) lets Genesis
    auto-pick the best available H.264 codec on the host.
    """

    filename: str = ""
    fps: float | None = None
    codec: str = ""
    save_on_reset: bool = False
    env_idx: int = 0


@dataclass
class RecordingCfg:
    """One data source × N output sinks.

    ``source`` is either:

    * the name of a sensor registered on :class:`InteractiveSceneCfg.sensors` — combined
      with ``field`` to pull a tensor out of the sensor's :class:`Sensor.data` payload, or
    * a callable. The callable may take zero arguments or one positional argument (the
      live :class:`ManagerBasedRlEnv`); arity is inspected once at registration. ``field``
      must be ``None`` for callable sources.

    ``env_idx`` squeezes a multi-env tensor leading dimension before handing the value to
    each sink. Plotters need scalar / 1-D tuples; file writers accept anything. Set
    ``env_idx=None`` to skip the squeeze (e.g. when the source already returns a scalar,
    or when you want to dump the full batch to ``.npz``).
    """

    name: str
    source: str | Callable[..., Any] | None = None
    field: str | None = None
    env_idx: int | None = 0
    outputs: tuple[OutputCfg, ...] = dataclasses.field(default_factory=tuple)
