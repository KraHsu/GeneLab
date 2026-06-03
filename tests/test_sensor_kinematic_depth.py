"""Unit tests for :class:`genelab.sensor.KinematicDepthSensor` (Genesis 1.0 ContactDepthProbe wrapper)."""

import pytest

torch = pytest.importorskip("torch")

from tests._sensor_fakes import FakeArticulation, FakeGsScene  # noqa: E402

from genelab.sensor.kinematic_depth import (  # noqa: E402
    KinematicDepthSensor,
    KinematicDepthSensorCfg,
)


def _build(
    *,
    link_name: str = "tip",
    probe_local_pos: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),),
    probe_radius: float = 0.01,
    probe_radius_noise: float = 0.0,
    history_length: int = 0,
) -> tuple[KinematicDepthSensor, FakeGsScene, dict[str, FakeArticulation]]:
    cfg = KinematicDepthSensorCfg(
        name="kd",
        link_name=link_name,
        probe_local_pos=probe_local_pos,
        probe_radius=probe_radius,
        probe_radius_noise=probe_radius_noise,
        history_length=history_length,
    )
    sensor = cfg.build()
    gs_scene = FakeGsScene(num_envs=2)
    entities = {"robot": FakeArticulation(link_names=["base", "tip"])}
    sensor.pre_build_genesis(gs_scene, entities)
    return sensor, gs_scene, entities


def test_pre_build_registers_contact_depth_probe_options() -> None:
    sensor, gs_scene, _ = _build(
        probe_local_pos=((0.1, 0.0, 0.0), (-0.1, 0.0, 0.0)),
        probe_radius=0.02,
        probe_radius_noise=5e-4,
        history_length=2,
    )
    assert len(gs_scene.sensors) == 1
    opts = gs_scene.sensors[0].opts
    assert opts.entity_idx == 0
    assert opts.link_idx_local == 1
    assert tuple(opts.probe_local_pos) == ((0.1, 0.0, 0.0), (-0.1, 0.0, 0.0))
    assert opts.probe_radius == pytest.approx(0.02)
    assert opts.probe_radius_noise == pytest.approx(5e-4)
    assert opts.history_length == 2
    del sensor


def test_empty_link_name_raises() -> None:
    cfg = KinematicDepthSensorCfg(name="kd", link_name="")
    with pytest.raises(ValueError, match="requires link_name"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["base"])}
        )


def test_unknown_link_name_raises() -> None:
    cfg = KinematicDepthSensorCfg(name="kd", link_name="nope")
    with pytest.raises(ValueError, match="not in link_names"):
        cfg.build().pre_build_genesis(
            FakeGsScene(), {"robot": FakeArticulation(link_names=["base", "tip"])}
        )


def test_compute_data_returns_genesis_depth_tensor() -> None:
    sensor, gs_scene, _ = _build(probe_local_pos=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)))
    # Genesis ``ContactDepthProbe.read()`` returns a ``(B, P)`` float depth tensor (metres).
    expected = torch.tensor([[0.01, 0.0], [0.0, 0.0]])
    gs_scene.sensors[0].set_return(expected)
    data = sensor.data
    assert data.depth.dtype == torch.float32
    assert torch.equal(data.depth, expected)
    # ``raw`` aliases ``depth`` so the generic reward primitives see the right tensor.
    assert torch.equal(data.raw, expected)
    # Re-read after an explicit invalidate triggers another read.
    sensor.update(0.02)
    _ = sensor.data
    assert gs_scene.sensors[0].read_calls == 2


def test_read_before_pre_build_asserts() -> None:
    sensor = KinematicDepthSensorCfg(name="kd", link_name="tip").build()
    with pytest.raises(AssertionError, match="read before pre_build_genesis"):
        _ = sensor.data
