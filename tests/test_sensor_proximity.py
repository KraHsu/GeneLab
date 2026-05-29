"""Unit tests for :class:`genelab.sensor.ProximitySensor` (Genesis 1.0 SurfaceDistanceProbe wrapper)."""

import pytest

torch = pytest.importorskip("torch")

from tests._sensor_fakes import FakeArticulation, FakeGsScene  # noqa: E402

from genelab.sensor.proximity import ProximitySensor, ProximitySensorCfg  # noqa: E402


def _build(
    *,
    link_name: str = "foot",
    probe_local_pos: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),),
    probe_radius: float = 0.5,
    track_link_names: tuple[str, ...] = ("terrain",),
    history_length: int = 0,
) -> tuple[ProximitySensor, FakeGsScene, dict[str, FakeArticulation]]:
    cfg = ProximitySensorCfg(
        name="prox",
        link_name=link_name,
        probe_local_pos=probe_local_pos,
        probe_radius=probe_radius,
        track_link_names=track_link_names,
        history_length=history_length,
    )
    sensor = cfg.build()
    gs_scene = FakeGsScene(num_envs=2)
    entities = {"robot": FakeArticulation(link_names=["base", "foot", "terrain"])}
    sensor.pre_build_genesis(gs_scene, entities)
    return sensor, gs_scene, entities


def test_pre_build_registers_surface_distance_probe_options() -> None:
    sensor, gs_scene, _ = _build(
        probe_local_pos=((0.05, 0.0, -0.02),),
        probe_radius=0.3,
        track_link_names=("terrain",),
        history_length=1,
    )
    assert len(gs_scene.sensors) == 1
    opts = gs_scene.sensors[0].opts
    assert opts.entity_idx == 0
    assert opts.link_idx_local == 1
    assert tuple(opts.probe_local_pos) == ((0.05, 0.0, -0.02),)
    assert opts.probe_radius == pytest.approx(0.3)
    assert tuple(opts.track_link_idx) == (2,)
    assert opts.history_length == 1
    del sensor


def test_empty_track_link_names_raises() -> None:
    cfg = ProximitySensorCfg(name="prox", link_name="foot", track_link_names=())
    with pytest.raises(ValueError, match="track_link_name"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["base", "foot"])}
        )


def test_unknown_track_link_name_raises() -> None:
    cfg = ProximitySensorCfg(name="prox", link_name="foot", track_link_names=("ghost",))
    with pytest.raises(ValueError, match="track_link_names"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["base", "foot"])}
        )


def test_empty_link_name_raises() -> None:
    cfg = ProximitySensorCfg(name="prox", link_name="")
    with pytest.raises(ValueError, match="requires link_name"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["foot"])}
        )


def test_compute_data_wraps_read() -> None:
    sensor, gs_scene, _ = _build(probe_local_pos=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)))
    gs_scene.sensors[0].set_return(torch.tensor([[0.1, 0.5], [0.2, 0.5]]))
    data = sensor.data
    assert torch.equal(data.distance, torch.tensor([[0.1, 0.5], [0.2, 0.5]]))


def test_read_before_pre_build_asserts() -> None:
    sensor = ProximitySensorCfg(name="prox", link_name="foot").build()
    with pytest.raises(AssertionError, match="read before pre_build_genesis"):
        _ = sensor.data
