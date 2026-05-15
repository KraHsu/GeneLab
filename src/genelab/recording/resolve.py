"""Translate a :class:`RecordingCfg` into a nullary callable Genesis can sample.

A resolver returns a closure with the signature ``Callable[[], Any]`` (matches Genesis's
:meth:`Recorder._data_func`). The closure reads :attr:`RecorderBridge.sensors` /
:attr:`RecorderBridge.env` at call time, so it remains valid even though the env is bound
after :func:`register_recorders` runs.

The closure also handles multi-env squeezing: a sensor returning a ``(num_envs, ...)``
tensor is sliced to ``[env_idx]`` (default 0). Set ``RecordingCfg.env_idx=None`` to keep
the full batch (useful when dumping to NPZ).

Build-time probe: recent Genesis line plotters call ``data_func()`` once inside
:meth:`Recorder.build` (which runs at the tail of ``gs_scene.build``) to determine the
plot's subplot / channel layout. This happens *before* Genelab can bind the env or
sensors. The resolver therefore wraps every callable with a stub-aware shim: when
``bridge.env`` is ``None`` (the only time this is true is during the build probe), a
zero tensor of the right length — derived from the recording's plot ``labels`` — is
returned. Once the env is bound (by ``RecorderBridge.bind_env``), the real data path is
used.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Any

import torch

from genelab.recording.cfg import (
    CSVFileCfg,
    MPLPlotCfg,
    NPZFileCfg,
    OutputCfg,
    PyQtPlotCfg,
    RecordingCfg,
    VideoFileCfg,
)

if TYPE_CHECKING:
    from genelab.recording.bridge import RecorderBridge


_PLOTTER_CFGS = (PyQtPlotCfg, MPLPlotCfg)
_FILE_CFGS = (NPZFileCfg, CSVFileCfg)


def resolve_data_func(rec_cfg: RecordingCfg, bridge: "RecorderBridge") -> Callable[[], Any]:
    """Return a nullary callable that produces the sample value at each recorder tick.

    The returned closure handles two regimes:

    1. **Build-time probe** (``bridge.env is None``): Genesis line plotters call this
       once inside :meth:`Recorder.build` to learn the plot's channel layout. The
       closure returns a stub tensor / dict whose shape matches the recording's
       ``labels`` so :class:`LinePlotHelper` can configure its subplots without
       touching the not-yet-bound sensors.
    2. **Runtime** (``bridge.env`` is set): the real data path runs — sensor cache
       access, dotted ``field`` walk, multi-env squeeze.

    Validation that ``field`` is consistent with the source kind happens here so misuse
    fails fast at scene build time (before any physics step runs).
    """
    source = rec_cfg.source
    if source is None:
        raise ValueError(f"RecordingCfg(name={rec_cfg.name!r}): source is required")

    if callable(source):
        if rec_cfg.field is not None:
            raise ValueError(
                f"RecordingCfg(name={rec_cfg.name!r}): field=... is only valid with a "
                "string (sensor-name) source; drop it when source is a callable."
            )
        real = _resolve_callable(rec_cfg, bridge)
    else:
        # The dataclass type annotation is ``str | Callable | None``; with ``None`` and
        # callable filtered above the remaining branch is a sensor-name string.
        real = _resolve_sensor(rec_cfg, bridge)

    stub = _build_probe_stub(rec_cfg)

    def _probe_aware() -> Any:
        if bridge.env is None:
            return stub
        return real()

    return _probe_aware


def is_sensor_source(rec_cfg: RecordingCfg) -> bool:
    """True when the source is a sensor name string."""
    return isinstance(rec_cfg.source, str)


def validate_output_compatibility(rec_cfg: RecordingCfg, bridge: "RecorderBridge") -> None:
    """Reject combinations the resolver can't satisfy cleanly.

    Currently: a :class:`VideoFileCfg` output requires a camera-sensor source, and a
    camera sensor source cannot be paired with a plot output (frames are not 1-D).
    """
    if not rec_cfg.outputs:
        raise ValueError(f"RecordingCfg(name={rec_cfg.name!r}): outputs is empty")

    has_video = any(isinstance(o, VideoFileCfg) for o in rec_cfg.outputs)
    has_plot = any(isinstance(o, _PLOTTER_CFGS) for o in rec_cfg.outputs)
    sensor = bridge.sensors.get(rec_cfg.source) if isinstance(rec_cfg.source, str) else None
    is_camera = sensor is not None and type(sensor).__name__ == "CameraSensor"

    if has_video and not is_camera:
        raise ValueError(
            f"RecordingCfg(name={rec_cfg.name!r}): VideoFileCfg requires a CameraSensor "
            f"source; got source={rec_cfg.source!r}"
        )
    if is_camera and has_plot:
        raise ValueError(
            f"RecordingCfg(name={rec_cfg.name!r}): plot outputs cannot consume camera "
            "frames; split into a separate RecordingCfg or remove the plot output."
        )


def _resolve_callable(rec_cfg: RecordingCfg, bridge: "RecorderBridge") -> Callable[[], Any]:
    source = rec_cfg.source
    assert callable(source)
    arity = _callable_arity(source)
    env_idx = rec_cfg.env_idx

    if arity == 0:

        def _get_zero() -> Any:
            value = source()
            return _maybe_squeeze(value, env_idx, bridge)

        return _get_zero

    def _get_env() -> Any:
        env = bridge.env
        if env is None:
            raise RuntimeError(
                f"RecordingCfg(name={rec_cfg.name!r}): data func invoked before "
                "RecorderBridge.bind_env; this is a Genelab internal ordering bug."
            )
        value = source(env)
        return _maybe_squeeze(value, env_idx, bridge)

    return _get_env


def _resolve_sensor(rec_cfg: RecordingCfg, bridge: "RecorderBridge") -> Callable[[], Any]:
    name = rec_cfg.source
    assert isinstance(name, str)
    field_path = rec_cfg.field
    env_idx = rec_cfg.env_idx

    # Camera→video special case: the sole output is VideoFileCfg, source resolves to a
    # CameraSensor — bypass the generic dataclass-payload error path and dive straight
    # into ``cam.data.rgb``.
    if len(rec_cfg.outputs) == 1 and isinstance(rec_cfg.outputs[0], VideoFileCfg):
        return _camera_video_getter(rec_cfg, bridge)

    def _get() -> Any:
        sensor = bridge.sensors.get(name)
        if sensor is None:
            raise KeyError(
                f"RecordingCfg(name={rec_cfg.name!r}): sensor {name!r} not found "
                f"in scene.sensors; registered names: {sorted(bridge.sensors)!r}"
            )
        data = sensor.data
        if field_path is None:
            if is_dataclass(data):
                raise ValueError(
                    f"RecordingCfg(name={rec_cfg.name!r}): sensor {name!r} returns a "
                    f"dataclass ({type(data).__name__}); specify field=<attr> to pick "
                    "one of its tensors."
                )
            value = data
        else:
            value = _walk_attr(data, field_path)
        return _maybe_squeeze(value, env_idx, bridge)

    return _get


def _camera_video_getter(rec_cfg: RecordingCfg, bridge: "RecorderBridge") -> Callable[[], Any]:
    video_cfg = rec_cfg.outputs[0]
    assert isinstance(video_cfg, VideoFileCfg)
    env_idx = video_cfg.env_idx
    sensor_name = rec_cfg.source
    assert isinstance(sensor_name, str)

    def _get() -> Any:
        sensor = bridge.sensors.get(sensor_name)
        if sensor is None:
            raise KeyError(
                f"RecordingCfg(name={rec_cfg.name!r}): camera sensor {sensor_name!r} "
                "not found in scene.sensors"
            )
        data = sensor.data
        rgb = getattr(data, "rgb", None)
        if rgb is None:
            raise RuntimeError(
                f"RecordingCfg(name={rec_cfg.name!r}): camera {sensor_name!r} has "
                "rgb=None; set render_rgb=True on its CameraSensorCfg."
            )
        if isinstance(rgb, torch.Tensor) and rgb.dim() >= 1:
            return rgb[env_idx]
        return rgb

    return _get


def _walk_attr(obj: Any, dotted: str) -> Any:
    """Walk ``a.b.c`` style attribute paths."""
    cur = obj
    for part in dotted.split("."):
        if not hasattr(cur, part):
            raise AttributeError(
                f"path {dotted!r}: object {type(cur).__name__} has no attribute {part!r}"
            )
        cur = getattr(cur, part)
    return cur


def _maybe_squeeze(value: Any, env_idx: int | None, bridge: "RecorderBridge") -> Any:
    if env_idx is None:
        return value
    if not isinstance(value, torch.Tensor) or value.dim() == 0:
        return value
    num_envs = bridge.env.num_envs if bridge.env is not None else None
    if num_envs is None or value.shape[0] != num_envs:
        # Not a per-env tensor — pass through.
        return value
    return value[env_idx]


def _callable_arity(fn: Callable[..., Any]) -> int:
    """Count required positional parameters (treat ``*args`` / ``**kwargs`` as zero-arity)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # C builtins etc. — assume nullary, the call will raise at runtime if wrong.
        return 0
    count = 0
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.default is inspect.Parameter.empty and param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            count += 1
    return min(count, 1)


__all__ = [
    "is_sensor_source",
    "resolve_data_func",
    "validate_output_compatibility",
]


# Re-export helpers used by register.py
def output_is_file(output: OutputCfg) -> bool:
    return isinstance(output, _FILE_CFGS) or isinstance(output, VideoFileCfg)


def output_is_plot(output: OutputCfg) -> bool:
    return isinstance(output, _PLOTTER_CFGS)


def _build_probe_stub(rec_cfg: RecordingCfg) -> Any:
    """Synthesize the value the data func returns during Genesis's build-time probe.

    Line plotters need a sample at build to determine subplot / channel structure.
    Use the first plot output's ``labels`` to choose a stub shape:

    * dict labels ``{"force": ("fx", "fy", "fz"), "torque": ("tx", ...)}`` →
      ``{"force": zeros(3), "torque": zeros(3)}``
    * tuple labels ``("ax", "ay", "az")`` → ``zeros(3)``
    * no labels on any plot output → ``zeros(1)`` (single-channel scalar plot)
    * recording has no plot outputs → ``zeros(1)`` (cheap; file writers ignore the
      shape at build time anyway)
    """
    plot_labels = None
    for out in rec_cfg.outputs:
        if isinstance(out, _PLOTTER_CFGS) and out.labels is not None:
            plot_labels = out.labels
            break

    if plot_labels is None:
        return torch.zeros(1)
    if isinstance(plot_labels, dict):
        return {key: torch.zeros(len(labels)) for key, labels in plot_labels.items()}
    return torch.zeros(len(plot_labels))
