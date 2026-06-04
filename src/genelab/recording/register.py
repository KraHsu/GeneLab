"""Translate :class:`RecordingCfg` objects into Genesis recorder registrations.

Called from :meth:`genelab.scene.InteractiveScene.build` after sensor pre-build and
before ``gs_scene.build`` (Genesis's ``add_recorder`` is ``@assert_unbuilt``). Each
``(RecordingCfg, OutputCfg)`` pair becomes one Genesis :class:`Recorder`; the bridge
collects the recorder handle so the env-side ``bind_env`` hook can patch its
``_steps_per_sample`` to the control-step decimation when appropriate.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from genelab.recording.cfg import (
    CSVFileCfg,
    MPLImagePlotCfg,
    MPLPlotCfg,
    NPZFileCfg,
    OutputCfg,
    PyQtPlotCfg,
    RecordingCfg,
    VideoFileCfg,
)
from genelab.recording.resolve import (
    is_sensor_source,
    resolve_data_func,
    validate_output_compatibility,
)

if TYPE_CHECKING:
    from genelab.recording.bridge import RecorderBridge


def register_recorders(
    *,
    gs_scene: Any,
    bridge: "RecorderBridge",
    recording_cfgs: tuple[RecordingCfg, ...],
    physics_dt: float,
) -> None:
    """Register every ``(RecordingCfg, OutputCfg)`` pair with ``gs_scene._recorder_manager``.

    Must run between sensor pre-build and ``gs_scene.build``. Default sampling rate is
    ``hz=None`` (every physics tick); the bridge later patches sensor-name sources to
    ``_steps_per_sample = env._decimation`` once decimation is known.
    """
    if not recording_cfgs:
        return

    from genelab.recording.bridge import RecorderHandle

    del physics_dt  # currently unused; bridge.bind_env handles rate via decimation

    for rec_cfg in recording_cfgs:
        validate_output_compatibility(rec_cfg, bridge)
        data_func = resolve_data_func(rec_cfg, bridge)
        sensor_src = is_sensor_source(rec_cfg)
        for output_cfg in rec_cfg.outputs:
            user_set_hz = output_cfg.hz is not None
            gs_options = _to_genesis_options(rec_cfg, output_cfg)
            recorder = gs_scene.start_recording(data_func, gs_options)
            if recorder is None:
                raise RuntimeError(
                    f"RecordingCfg(name={rec_cfg.name!r}): gs_scene.start_recording "
                    "returned None; Genesis API contract violated."
                )
            bridge.handles.append(
                RecorderHandle(
                    recorder=recorder,
                    is_sensor_source=sensor_src,
                    user_set_hz=user_set_hz,
                    save_on_reset=_save_on_reset(output_cfg),
                )
            )


def _save_on_reset(output_cfg: OutputCfg) -> bool:
    return bool(getattr(output_cfg, "save_on_reset", False))


def _to_genesis_options(rec_cfg: RecordingCfg, output_cfg: OutputCfg) -> Any:
    """Build a Genesis ``RecorderOptions`` instance matching ``output_cfg``."""
    from genesis import recorders as gs_recorders

    common: dict[str, Any] = {
        "hz": output_cfg.hz,
        "buffer_size": output_cfg.buffer_size,
        "buffer_full_wait_time": output_cfg.buffer_full_wait_time,
    }
    title_default = rec_cfg.name

    if isinstance(output_cfg, PyQtPlotCfg):
        return gs_recorders.PyQtLinePlot(
            **common,
            title=output_cfg.title or title_default,
            labels=output_cfg.labels,
            window_size=output_cfg.window_size,
            history_length=output_cfg.history_length,
        )
    if isinstance(output_cfg, MPLPlotCfg):
        return gs_recorders.MPLLinePlot(
            **common,
            title=output_cfg.title or title_default,
            labels=output_cfg.labels,
            window_size=output_cfg.window_size,
            history_length=output_cfg.history_length,
        )
    if isinstance(output_cfg, MPLImagePlotCfg):
        return gs_recorders.MPLImagePlot(
            **common,
            title=output_cfg.title or title_default,
            window_size=output_cfg.window_size,
        )
    if isinstance(output_cfg, NPZFileCfg):
        if not output_cfg.filename:
            raise ValueError(
                f"RecordingCfg(name={rec_cfg.name!r}): NPZFileCfg.filename is required"
            )
        return gs_recorders.NPZFile(
            **common,
            filename=output_cfg.filename,
            save_on_reset=output_cfg.save_on_reset,
        )
    if isinstance(output_cfg, CSVFileCfg):
        if not output_cfg.filename:
            raise ValueError(
                f"RecordingCfg(name={rec_cfg.name!r}): CSVFileCfg.filename is required"
            )
        return gs_recorders.CSVFile(
            **common,
            filename=output_cfg.filename,
            header=output_cfg.header,
            save_every_write=output_cfg.save_every_write,
            save_on_reset=output_cfg.save_on_reset,
        )
    if isinstance(output_cfg, VideoFileCfg):
        if not output_cfg.filename:
            raise ValueError(
                f"RecordingCfg(name={rec_cfg.name!r}): VideoFileCfg.filename is required"
            )
        kwargs: dict[str, Any] = {
            "filename": output_cfg.filename,
            "save_on_reset": output_cfg.save_on_reset,
        }
        if output_cfg.codec:
            kwargs["codec"] = output_cfg.codec
        if output_cfg.fps is not None:
            kwargs["fps"] = output_cfg.fps
        return gs_recorders.VideoFile(**common, **kwargs)
    raise TypeError(
        f"RecordingCfg(name={rec_cfg.name!r}): unknown OutputCfg type {type(output_cfg).__name__}"
    )


# Re-export so the call-site can construct handles without importing the class directly.
_register_recorders: Callable[..., None] = register_recorders
__all__ = ["register_recorders"]
