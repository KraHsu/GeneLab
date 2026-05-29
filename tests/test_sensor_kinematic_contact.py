"""Unit tests for :class:`genelab.sensor.KinematicContactSensor` (Genesis 1.0 ContactProbe wrapper)."""

import pytest

torch = pytest.importorskip("torch")

from tests._sensor_fakes import FakeArticulation, FakeGsScene  # noqa: E402

from genelab.sensor.kinematic_contact import (  # noqa: E402
    KinematicContactSensor,
    KinematicContactSensorCfg,
)


def _build(
    *,
    link_name: str = "tip",
    probe_local_pos: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),),
    contact_threshold: float = 1e-4,
    history_length: int = 0,
) -> tuple[KinematicContactSensor, FakeGsScene, dict[str, FakeArticulation]]:
    cfg = KinematicContactSensorCfg(
        name="kc",
        link_name=link_name,
        probe_local_pos=probe_local_pos,
        contact_threshold=contact_threshold,
        history_length=history_length,
    )
    sensor = cfg.build()
    gs_scene = FakeGsScene(num_envs=2)
    entities = {"robot": FakeArticulation(link_names=["base", "tip"])}
    sensor.pre_build_genesis(gs_scene, entities)
    return sensor, gs_scene, entities


def test_pre_build_registers_contact_probe_options() -> None:
    sensor, gs_scene, _ = _build(
        probe_local_pos=((0.1, 0.0, 0.0), (-0.1, 0.0, 0.0)),
        contact_threshold=5e-4,
        history_length=2,
    )
    assert len(gs_scene.sensors) == 1
    opts = gs_scene.sensors[0].opts
    assert opts.entity_idx == 0
    assert opts.link_idx_local == 1
    assert tuple(opts.probe_local_pos) == ((0.1, 0.0, 0.0), (-0.1, 0.0, 0.0))
    assert opts.contact_threshold == pytest.approx(5e-4)
    assert opts.history_length == 2
    del sensor  # silence


def test_empty_link_name_raises() -> None:
    cfg = KinematicContactSensorCfg(name="kc", link_name="")
    with pytest.raises(ValueError, match="requires link_name"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["base"])}
        )


def test_unknown_link_name_raises() -> None:
    cfg = KinematicContactSensorCfg(name="kc", link_name="nope")
    with pytest.raises(ValueError, match="not in link_names"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["base", "tip"])}
        )


def test_compute_data_thresholds_depth_to_in_contact() -> None:
    sensor, gs_scene, _ = _build(
        probe_local_pos=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
        contact_threshold=1e-3,
    )
    # Two probes per env; first probe in contact (depth > threshold), second not.
    gs_scene.sensors[0].set_return(torch.tensor([[2e-3, 5e-5], [4e-3, 8e-4]]))
    data = sensor.data
    assert torch.equal(data.depth, torch.tensor([[2e-3, 5e-5], [4e-3, 8e-4]]))
    assert torch.equal(data.in_contact, torch.tensor([[True, False], [True, False]]))
    # Re-read after an explicit invalidate triggers another read.
    sensor.update(0.02)
    _ = sensor.data
    assert gs_scene.sensors[0].read_calls == 2


def test_read_before_pre_build_asserts() -> None:
    sensor = KinematicContactSensorCfg(name="kc", link_name="tip").build()
    with pytest.raises(AssertionError, match="read before pre_build_genesis"):
        _ = sensor.data
