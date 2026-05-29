"""Unit tests for :class:`genelab.sensor.ElastomerTactileSensor` (Genesis 1.0 ElastomerTaxel wrapper)."""

import pytest

torch = pytest.importorskip("torch")

from tests._sensor_fakes import FakeArticulation, FakeGsScene  # noqa: E402

from genelab.sensor.tactile_elastomer import (  # noqa: E402
    ElastomerTactileSensor,
    ElastomerTactileSensorCfg,
)


def _build(
    *,
    link_name: str = "fingertip",
    probe_local_pos: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),),
    track_link_names: tuple[str, ...] = ("peg",),
    history_length: int = 0,
    lambda_d: float = 700.0,
    lambda_s: float = 300.0,
    n_sample_points: int = 500,
) -> tuple[ElastomerTactileSensor, FakeGsScene, dict[str, FakeArticulation]]:
    cfg = ElastomerTactileSensorCfg(
        name="elastomer",
        link_name=link_name,
        probe_local_pos=probe_local_pos,
        track_link_names=track_link_names,
        lambda_d=lambda_d,
        lambda_s=lambda_s,
        n_sample_points=n_sample_points,
        history_length=history_length,
    )
    sensor = cfg.build()
    gs_scene = FakeGsScene(num_envs=2)
    entities = {"robot": FakeArticulation(link_names=["base", "fingertip", "peg"])}
    sensor.pre_build_genesis(gs_scene, entities)
    return sensor, gs_scene, entities


def test_pre_build_forwards_elastomer_options() -> None:
    sensor, gs_scene, _ = _build(
        probe_local_pos=((0.0, 0.0, 0.005),),
        track_link_names=("peg",),
        lambda_d=600.0,
        lambda_s=250.0,
        n_sample_points=128,
        history_length=2,
    )
    opts = gs_scene.sensors[0].opts
    assert opts.entity_idx == 0
    assert opts.link_idx_local == 1
    assert tuple(opts.probe_local_pos) == ((0.0, 0.0, 0.005),)
    assert tuple(opts.track_link_idx) == (2,)
    assert opts.lambda_d == pytest.approx(600.0)
    assert opts.lambda_s == pytest.approx(250.0)
    assert opts.n_sample_points == 128
    assert opts.history_length == 2
    del sensor


def test_empty_track_link_names_raises() -> None:
    cfg = ElastomerTactileSensorCfg(name="elastomer", link_name="fingertip", track_link_names=())
    with pytest.raises(ValueError, match="track_link_name"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["base", "fingertip"])}
        )


def test_unknown_track_link_name_raises() -> None:
    cfg = ElastomerTactileSensorCfg(
        name="elastomer", link_name="fingertip", track_link_names=("ghost",)
    )
    with pytest.raises(ValueError, match="track_link_names"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["base", "fingertip"])}
        )


def test_empty_link_name_raises() -> None:
    cfg = ElastomerTactileSensorCfg(name="elastomer", link_name="")
    with pytest.raises(ValueError, match="requires link_name"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["fingertip"])}
        )


def test_compute_data_wraps_read() -> None:
    sensor, gs_scene, _ = _build()
    expected = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.1, 0.0, -0.02]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    gs_scene.sensors[0].set_return(expected)
    data = sensor.data
    assert torch.equal(data.raw, expected)


def test_read_before_pre_build_asserts() -> None:
    sensor = ElastomerTactileSensorCfg(
        name="elastomer", link_name="fingertip", track_link_names=("peg",)
    ).build()
    with pytest.raises(AssertionError, match="read before pre_build_genesis"):
        _ = sensor.data
