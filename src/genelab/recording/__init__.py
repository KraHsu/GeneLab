"""Declarative recording / live-plotting layer over Genesis's :mod:`genesis.recorders`.

A :class:`RecordingCfg` placed on :class:`InteractiveSceneCfg.recordings` describes one
data source (sensor name + optional ``field``, or a custom callable) feeding one or more
output sinks: real-time PyQt/Matplotlib plots, NPZ/CSV/MP4 files. The
:class:`RecorderBridge` wires the data callables to the live env once it's constructed,
and the scene's build/close hooks drive the Genesis recorder lifecycle.

Typical usage::

    from genelab.recording import NPZFileCfg, PyQtPlotCfg, RecordingCfg

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
"""

from genelab.recording.bridge import RecorderBridge
from genelab.recording.cfg import (
    CSVFileCfg,
    MPLPlotCfg,
    NPZFileCfg,
    OutputCfg,
    PyQtPlotCfg,
    RecordingCfg,
    VideoFileCfg,
)

__all__ = [
    "CSVFileCfg",
    "MPLPlotCfg",
    "NPZFileCfg",
    "OutputCfg",
    "PyQtPlotCfg",
    "RecorderBridge",
    "RecordingCfg",
    "VideoFileCfg",
]
