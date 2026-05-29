"""Unit tests for :class:`genelab.sensor.TemperatureGridSensor` (Genesis 1.0 TemperatureGrid wrapper)."""

import pytest

torch = pytest.importorskip("torch")

from tests._sensor_fakes import FakeArticulation, FakeGsScene  # noqa: E402

from genelab.sensor.temperature import (  # noqa: E402
    TemperatureGridSensor,
    TemperatureGridSensorCfg,
)


def _build(
    *,
    link_name: str = "tip",
    grid_size: tuple[int, int, int] = (2, 2, 2),
    ambient_temperature: float | None = None,
    history_length: int = 0,
) -> tuple[TemperatureGridSensor, FakeGsScene, dict[str, FakeArticulation]]:
    cfg = TemperatureGridSensorCfg(
        name="t",
        link_name=link_name,
        grid_size=grid_size,
        ambient_temperature=ambient_temperature,
        history_length=history_length,
    )
    sensor = cfg.build()
    gs_scene = FakeGsScene(num_envs=2)
    entities = {"robot": FakeArticulation(link_names=["base", "tip"])}
    sensor.pre_build_genesis(gs_scene, entities)
    return sensor, gs_scene, entities


def test_pre_build_forwards_grid_and_ambient() -> None:
    sensor, gs_scene, _ = _build(grid_size=(3, 1, 2), ambient_temperature=22.5, history_length=4)
    assert len(gs_scene.sensors) == 1
    opts = gs_scene.sensors[0].opts
    assert opts.entity_idx == 0
    assert opts.link_idx_local == 1
    assert tuple(opts.grid_size) == (3, 1, 2)
    assert opts.ambient_temperature == pytest.approx(22.5)
    assert opts.history_length == 4
    del sensor


def test_ambient_temperature_none_omits_kwarg() -> None:
    _, gs_scene, _ = _build(ambient_temperature=None)
    # When ambient_temperature is None, the wrapper must not pass it through; the Genesis
    # default ``ambient_temperature=None`` then takes effect.
    opts = gs_scene.sensors[0].opts
    assert opts.ambient_temperature is None


def test_empty_link_name_raises() -> None:
    cfg = TemperatureGridSensorCfg(name="t", link_name="")
    with pytest.raises(ValueError, match="requires link_name"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["tip"])}
        )


def test_unknown_link_name_raises() -> None:
    cfg = TemperatureGridSensorCfg(name="t", link_name="nope")
    with pytest.raises(ValueError, match="not in link_names"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["tip"])}
        )


def test_compute_data_wraps_read() -> None:
    sensor, gs_scene, _ = _build(grid_size=(2, 1, 1))
    expected = torch.tensor([[[[20.0]], [[21.0]]], [[[19.5]], [[22.0]]]])  # (2, 2, 1, 1)
    gs_scene.sensors[0].set_return(expected)
    data = sensor.data
    assert torch.equal(data.temperature, expected)


def test_read_before_pre_build_asserts() -> None:
    sensor = TemperatureGridSensorCfg(name="t", link_name="tip").build()
    with pytest.raises(AssertionError, match="read before pre_build_genesis"):
        _ = sensor.data
