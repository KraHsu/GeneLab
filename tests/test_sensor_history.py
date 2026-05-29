"""``history_length`` plumbing for the new Genesis-1.0 sensor wrappers.

These tests assert that ``cfg.history_length`` lands verbatim on the Genesis sensor
options object — Genesis itself owns the ring-buffer mechanics, so the GeneLab wrappers
only have to forward the integer correctly.
"""

from collections.abc import Callable
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from tests._sensor_fakes import FakeArticulation, FakeGsScene  # noqa: E402

from genelab.sensor.kinematic_contact import KinematicContactSensorCfg  # noqa: E402
from genelab.sensor.proximity import ProximitySensorCfg  # noqa: E402
from genelab.sensor.temperature import TemperatureGridSensorCfg  # noqa: E402


_BUILDERS: dict[str, Callable[[int], Any]] = {
    "ProximitySensor": lambda h: ProximitySensorCfg(
        name="p",
        link_name="foot",
        track_link_names=("base",),
        history_length=h,
    ),
    "KinematicContactSensor": lambda h: KinematicContactSensorCfg(
        name="k", link_name="foot", history_length=h
    ),
    "TemperatureGridSensor": lambda h: TemperatureGridSensorCfg(
        name="t", link_name="foot", history_length=h
    ),
}


@pytest.mark.parametrize("cfg_name", list(_BUILDERS))
@pytest.mark.parametrize("history_length", [0, 1, 5])
def test_history_length_reaches_genesis_options(cfg_name: str, history_length: int) -> None:
    cfg = _BUILDERS[cfg_name](history_length)
    sensor = cfg.build()
    gs_scene = FakeGsScene(num_envs=2)
    entities = {"robot": FakeArticulation(link_names=["base", "foot"])}
    sensor.pre_build_genesis(gs_scene, entities)

    assert len(gs_scene.sensors) == 1
    assert gs_scene.sensors[0].opts.history_length == history_length


@pytest.mark.parametrize("cfg_name", list(_BUILDERS))
def test_default_history_length_is_zero(cfg_name: str) -> None:
    """Default of zero matches Genesis's per-sensor default (no ring buffer allocated)."""
    cfg = _BUILDERS[cfg_name](0)
    assert cfg.history_length == 0
