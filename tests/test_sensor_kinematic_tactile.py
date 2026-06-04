"""Unit tests for :class:`genelab.sensor.KinematicTactileSensor` (Genesis 1.0 KinematicTaxel wrapper)."""

import pytest

torch = pytest.importorskip("torch")

from tests._sensor_fakes import FakeArticulation, FakeGsScene  # noqa: E402

from genelab.sensor.kinematic_tactile import (  # noqa: E402
    KinematicTactileSensor,
    KinematicTactileSensorCfg,
)


def _build(
    *,
    link_name: str = "tip",
    probe_local_pos: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),),
    probe_local_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    normal_stiffness: float = 1000.0,
    shear_scalar: float = 1.0,
    history_length: int = 0,
) -> tuple[KinematicTactileSensor, FakeGsScene, dict[str, FakeArticulation]]:
    cfg = KinematicTactileSensorCfg(
        name="kt",
        link_name=link_name,
        probe_local_pos=probe_local_pos,
        probe_local_normal=probe_local_normal,
        normal_stiffness=normal_stiffness,
        shear_scalar=shear_scalar,
        history_length=history_length,
    )
    sensor = cfg.build()
    gs_scene = FakeGsScene(num_envs=2)
    entities = {"robot": FakeArticulation(link_names=["base", "tip"])}
    sensor.pre_build_genesis(gs_scene, entities)
    return sensor, gs_scene, entities


def test_pre_build_forwards_kinematic_taxel_options() -> None:
    sensor, gs_scene, _ = _build(
        probe_local_pos=((0.1, 0.0, 0.0), (-0.1, 0.0, 0.0)),
        probe_local_normal=(0.0, 1.0, 0.0),
        normal_stiffness=500.0,
        shear_scalar=0.25,
        history_length=3,
    )
    assert len(gs_scene.sensors) == 1
    opts = gs_scene.sensors[0].opts
    assert opts.entity_idx == 0
    assert opts.link_idx_local == 1
    assert tuple(opts.probe_local_pos) == ((0.1, 0.0, 0.0), (-0.1, 0.0, 0.0))
    assert tuple(opts.probe_local_normal) == (0.0, 1.0, 0.0)
    assert opts.normal_stiffness == pytest.approx(500.0)
    assert opts.shear_scalar == pytest.approx(0.25)
    assert opts.history_length == 3
    del sensor


def test_empty_link_name_raises() -> None:
    cfg = KinematicTactileSensorCfg(name="kt", link_name="")
    with pytest.raises(ValueError, match="requires link_name"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["base"])}
        )


def test_unknown_link_name_raises() -> None:
    cfg = KinematicTactileSensorCfg(name="kt", link_name="nope")
    with pytest.raises(ValueError, match="not in link_names"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["base", "tip"])}
        )


def test_compute_data_extracts_force_and_torque_fields() -> None:
    """Genesis ``KinematicTaxel.read()`` returns ``KinematicTaxelData(force=…, torque=…)``;
    the wrapper extracts both tensors and aliases ``raw`` to ``force`` for the generic
    reward primitives."""

    class _FakeKinematicTaxelData:
        def __init__(self, force: torch.Tensor, torque: torch.Tensor) -> None:
            self.force = force
            self.torque = torque

    sensor, gs_scene, _ = _build(probe_local_pos=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)))
    force = torch.tensor([[[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    torque = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    gs_scene.sensors[0].set_return(_FakeKinematicTaxelData(force, torque))  # type: ignore[arg-type]
    data = sensor.data
    assert torch.equal(data.force, force)
    assert torch.equal(data.torque, torque)
    # ``raw`` aliases ``force`` so the generic reward primitives still see the right tensor.
    assert torch.equal(data.raw, force)


def test_read_before_pre_build_asserts() -> None:
    sensor = KinematicTactileSensorCfg(name="kt", link_name="tip").build()
    with pytest.raises(AssertionError, match="read before pre_build_genesis"):
        _ = sensor.data
