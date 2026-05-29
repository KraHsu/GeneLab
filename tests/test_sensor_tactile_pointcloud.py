"""Unit tests for :class:`genelab.sensor.PointCloudTactileSensor` (Genesis 1.0 ProximityTaxel wrapper)."""

import pytest

torch = pytest.importorskip("torch")

from tests._sensor_fakes import FakeArticulation, FakeGsScene  # noqa: E402

from genelab.sensor.tactile_pointcloud import (  # noqa: E402
    PointCloudTactileSensor,
    PointCloudTactileSensorCfg,
)


def _build(
    *,
    link_name: str = "fingertip",
    probe_local_pos: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),),
    track_link_names: tuple[str, ...] = ("object",),
    history_length: int = 0,
    stiffness: float = 100.0,
    shear_coupling: float = 0.0,
    n_sample_points: int = 500,
) -> tuple[PointCloudTactileSensor, FakeGsScene, dict[str, FakeArticulation]]:
    cfg = PointCloudTactileSensorCfg(
        name="pcl",
        link_name=link_name,
        probe_local_pos=probe_local_pos,
        track_link_names=track_link_names,
        stiffness=stiffness,
        shear_coupling=shear_coupling,
        n_sample_points=n_sample_points,
        history_length=history_length,
    )
    sensor = cfg.build()
    gs_scene = FakeGsScene(num_envs=2)
    entities = {"robot": FakeArticulation(link_names=["base", "fingertip", "object"])}
    sensor.pre_build_genesis(gs_scene, entities)
    return sensor, gs_scene, entities


def test_pre_build_forwards_pointcloud_options() -> None:
    sensor, gs_scene, _ = _build(
        probe_local_pos=((0.0, 0.0, 0.001),),
        track_link_names=("object",),
        stiffness=250.0,
        shear_coupling=0.5,
        n_sample_points=64,
        history_length=3,
    )
    opts = gs_scene.sensors[0].opts
    assert opts.entity_idx == 0
    assert opts.link_idx_local == 1
    assert tuple(opts.probe_local_pos) == ((0.0, 0.0, 0.001),)
    assert tuple(opts.track_link_idx) == (2,)
    assert opts.stiffness == pytest.approx(250.0)
    assert opts.shear_coupling == pytest.approx(0.5)
    assert opts.n_sample_points == 64
    assert opts.history_length == 3
    del sensor


def test_empty_track_link_names_raises() -> None:
    cfg = PointCloudTactileSensorCfg(name="pcl", link_name="fingertip", track_link_names=())
    with pytest.raises(ValueError, match="track_link_name"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["base", "fingertip"])}
        )


def test_unknown_track_link_name_raises() -> None:
    cfg = PointCloudTactileSensorCfg(name="pcl", link_name="fingertip", track_link_names=("ghost",))
    with pytest.raises(ValueError, match="track_link_names"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["base", "fingertip"])}
        )


def test_empty_link_name_raises() -> None:
    cfg = PointCloudTactileSensorCfg(name="pcl", link_name="")
    with pytest.raises(ValueError, match="requires link_name"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["fingertip"])}
        )


def test_compute_data_extracts_force_and_torque_fields() -> None:
    """Genesis ``ProximityTaxel.read()`` returns ``ProximityTaxelData(force=…, torque=…)``;
    the wrapper extracts both tensors and aliases ``raw`` to ``force`` for the generic
    reward primitives."""

    class _FakeProximityTaxelData:
        def __init__(self, force: torch.Tensor, torque: torch.Tensor) -> None:
            self.force = force
            self.torque = torque

    sensor, gs_scene, _ = _build(probe_local_pos=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)))
    force = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    torque = torch.tensor([[[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    gs_scene.sensors[0].set_return(_FakeProximityTaxelData(force, torque))  # type: ignore[arg-type]
    data = sensor.data
    assert torch.equal(data.force, force)
    assert torch.equal(data.torque, torque)
    # ``raw`` aliases ``force`` so the generic reward primitives still see the right tensor.
    assert torch.equal(data.raw, force)


def test_read_before_pre_build_asserts() -> None:
    sensor = PointCloudTactileSensorCfg(
        name="pcl", link_name="fingertip", track_link_names=("object",)
    ).build()
    with pytest.raises(AssertionError, match="read before pre_build_genesis"):
        _ = sensor.data
