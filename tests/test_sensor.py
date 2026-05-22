"""Sensor abstraction + obs-noise pipeline. Uses a fake env so Genesis is not required."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from genelab.managers import (
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
)
from genelab.mdp.noise import BiasDrift, CorrelatedNoise, Gnoise, ScaledNoise, Unoise
from genelab.mdp.observations import joint_force_torque
from genelab.sensor import (
    BodyVelocitySensorCfg,
    CameraSensorCfg,
    ContactSensorCfg,
    ForceTorqueSensorCfg,
    FrameTransformerSensorCfg,
    GridPattern,
    HemispherePattern,
    IMUSensorCfg,
    RayCastSensorCfg,
    RingPattern,
    RootAngularMomentumSensorCfg,
    SelfContactSensorCfg,
    Sensor,
    SensorCfg,
    TargetFrameCfg,
    TerrainHeightSensorCfg,
)


@dataclass
class _ConstSensorCfg(SensorCfg):
    value: float = 1.0

    def build(self) -> "_ConstSensor":
        return _ConstSensor(self)


class _ConstSensor(Sensor[torch.Tensor]):
    def __init__(self, cfg: _ConstSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self.compute_calls: int = 0

    def _compute_data(self) -> torch.Tensor:
        assert self._env is not None
        self.compute_calls += 1
        return torch.full((self._env.num_envs, 3), self._cfg_typed.value, device=self._env.device)


class _FakeEnv:
    """Just enough surface for sensors and the observation manager."""

    def __init__(self, num_envs: int = 4, device: str = "cpu") -> None:
        self.num_envs = num_envs
        self.device = device
        self.sensors: dict[str, Sensor[Any]] = {}


def test_sensor_caches_until_invalidated() -> None:
    env = _FakeEnv(num_envs=2)
    sensor = _ConstSensorCfg(name="c", value=1.5).build()
    sensor.bind(env)
    first = sensor.data
    assert first.shape == (2, 3)
    assert torch.allclose(first, torch.full((2, 3), 1.5))
    assert sensor.compute_calls == 1
    _ = sensor.data
    assert sensor.compute_calls == 1
    sensor.update(0.02)
    _ = sensor.data
    assert sensor.compute_calls == 2
    sensor.reset(torch.tensor([0, 1]))
    _ = sensor.data
    assert sensor.compute_calls == 3


def test_unoise_stays_within_bounds_and_is_additive() -> None:
    torch.manual_seed(0)
    data = torch.zeros(1024, 3)
    noisy = Unoise(n_min=-0.2, n_max=0.2).apply(data)
    delta = noisy - data
    assert (delta >= -0.2 - 1e-6).all()
    assert (delta <= 0.2 + 1e-6).all()
    assert delta.abs().mean() > 1e-3


def test_gnoise_has_expected_std() -> None:
    torch.manual_seed(0)
    data = torch.zeros(4096, 1)
    noisy = Gnoise(mean=0.0, std=0.5).apply(data)
    assert abs(noisy.std().item() - 0.5) < 0.05


def test_scaled_noise_is_multiplicative_and_signal_proportional() -> None:
    # Degenerate range → exact ×1 (no corruption).
    data = torch.tensor([[2.0, -4.0]])
    assert torch.allclose(ScaledNoise(0.0, 0.0).apply(data), data)
    # Fixed +50% gain → data × 1.5; the perturbation scales with the signal.
    assert torch.allclose(ScaledNoise(0.5, 0.5).apply(data), data * 1.5)


def test_correlated_noise_zero_std_is_identity_and_keeps_state() -> None:
    noise = CorrelatedNoise(std=0.0, alpha=0.9)
    data = torch.zeros(4, 3)
    for _ in range(3):
        assert torch.allclose(noise.apply(data), data)  # white=0 → state stays 0
    # Stationary std ≈ std for any alpha (sqrt(1-alpha^2) mixing).
    torch.manual_seed(0)
    corr = CorrelatedNoise(std=0.5, alpha=0.8)
    flat = torch.zeros(20000, 1)
    last = corr.apply(flat)
    for _ in range(50):  # let the AR process reach its stationary regime
        last = corr.apply(flat)
    assert abs(last.std().item() - 0.5) < 0.1


def test_bias_drift_accumulates_and_respects_max_bias() -> None:
    # No drift → identity, bias stays zero across steps.
    nodrift = BiasDrift(drift_std=0.0)
    data = torch.zeros(8, 2)
    for _ in range(3):
        assert torch.allclose(nodrift.apply(data), data)
    # Large drift, clamped: |bias| never exceeds max_bias.
    torch.manual_seed(0)
    drift = BiasDrift(drift_std=10.0, max_bias=0.1)
    flat = torch.zeros(1000, 1)
    for _ in range(20):
        out = drift.apply(flat)
    assert (out.abs() <= 0.1 + 1e-6).all()
    assert out.abs().mean() > 1e-3  # the bias actually moved off zero


def test_observation_pipeline_noise_only_when_corruption_enabled() -> None:
    env = _FakeEnv(num_envs=8)

    def const_two(_env: Any) -> torch.Tensor:
        return torch.full((_env.num_envs, 3), 2.0, device=_env.device)

    torch.manual_seed(0)
    cfg = {
        "policy": ObservationGroupCfg(
            enable_corruption=True,
            terms={"x": ObservationTermCfg(func=const_two, noise=Unoise(-0.1, 0.1))},
        ),
        "critic": ObservationGroupCfg(
            enable_corruption=False,
            terms={"x": ObservationTermCfg(func=const_two, noise=Unoise(-0.1, 0.1))},
        ),
    }
    mgr = ObservationManager(cfg, env)
    obs = mgr.compute()
    assert torch.equal(obs["critic"], torch.full((8, 3), 2.0))
    assert not torch.equal(obs["policy"], obs["critic"])
    assert ((obs["policy"] - 2.0).abs() <= 0.1 + 1e-6).all()


def test_observation_pipeline_applies_noise_scale_clip_in_order() -> None:
    env = _FakeEnv(num_envs=1)

    def const_ten(_env: Any) -> torch.Tensor:
        return torch.full((_env.num_envs, 1), 10.0, device=_env.device)

    # zero-magnitude noise so we can pin the math: 10 -> +0 noise -> *0.5 scale -> clip to [0, 3]
    cfg = {
        "policy": ObservationGroupCfg(
            enable_corruption=True,
            terms={
                "x": ObservationTermCfg(
                    func=const_ten,
                    noise=Unoise(0.0, 0.0),
                    scale=0.5,
                    clip=(0.0, 3.0),
                )
            },
        )
    }
    mgr = ObservationManager(cfg, env)
    obs = mgr.compute()
    assert torch.allclose(obs["policy"], torch.tensor([[3.0]]))


def test_observation_manager_skips_corruption_when_noise_unset() -> None:
    env = _FakeEnv(num_envs=2)

    def const_one(_env: Any) -> torch.Tensor:
        return torch.full((_env.num_envs, 2), 1.0, device=_env.device)

    cfg = {
        "g": ObservationGroupCfg(
            enable_corruption=True,
            terms={"x": ObservationTermCfg(func=const_one)},
        )
    }
    mgr = ObservationManager(cfg, env)
    obs = mgr.compute()
    assert torch.equal(obs["g"], torch.full((2, 2), 1.0))


# --------------------------------------------------------------------- BodyVelocitySensor


class _FakeRobotState:
    def __init__(self, num_envs: int, num_links: int, device: str) -> None:
        self.link_quat_w = torch.zeros(num_envs, num_links, 4, device=device)
        self.link_quat_w[..., 0] = 1.0
        self.link_lin_vel_w = torch.zeros(num_envs, num_links, 3, device=device)
        self.link_ang_vel_w = torch.zeros(num_envs, num_links, 3, device=device)


class _FakeRobotEnv:
    def __init__(self, num_envs: int = 2, link_names: tuple[str, ...] = ("pelvis",)) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.link_names = list(link_names)
        self.robot_state = _FakeRobotState(num_envs, len(link_names), self.device)


def test_body_velocity_sensor_gyro_rotates_to_body_frame() -> None:
    env = _FakeRobotEnv()
    # 90° rotation about +z: q = (cos45°, 0, 0, sin45°). World ω = +x → body ω = +y after inverse.
    s2 = 2.0**0.5 / 2
    env.robot_state.link_quat_w[:, 0] = torch.tensor([s2, 0.0, 0.0, s2])
    env.robot_state.link_ang_vel_w[:, 0] = torch.tensor([1.0, 0.0, 0.0])
    sensor = BodyVelocitySensorCfg(name="g", link_name="pelvis", measure="ang_vel").build()
    sensor.bind(env)
    out = sensor.data
    assert out.shape == (2, 3)
    assert torch.allclose(out[0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6)


def test_body_velocity_sensor_velocimeter_lever_arm_cancels() -> None:
    # Pure rotation ω = +z, site offset r = +x → v_site_world = ω × r = +y.
    # Then rotate into body frame (identity quat here) → still +y.
    env = _FakeRobotEnv()
    env.robot_state.link_ang_vel_w[:, 0] = torch.tensor([0.0, 0.0, 1.0])
    sensor = BodyVelocitySensorCfg(
        name="v", link_name="pelvis", offset=(1.0, 0.0, 0.0), measure="lin_vel"
    ).build()
    sensor.bind(env)
    out = sensor.data
    assert torch.allclose(out[0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_body_velocity_sensor_velocimeter_with_translation_only() -> None:
    # Pure translation, no rotation: world vel directly returned in body frame (identity quat).
    env = _FakeRobotEnv()
    env.robot_state.link_lin_vel_w[:, 0] = torch.tensor([2.0, -1.0, 0.5])
    sensor = BodyVelocitySensorCfg(
        name="v", link_name="pelvis", offset=(0.05, 0.0, -0.08), measure="lin_vel"
    ).build()
    sensor.bind(env)
    out = sensor.data
    assert torch.allclose(out[0], torch.tensor([2.0, -1.0, 0.5]), atol=1e-6)


def test_body_velocity_sensor_bias_randomizes_on_reset() -> None:
    torch.manual_seed(0)
    env = _FakeRobotEnv(num_envs=64)
    sensor = BodyVelocitySensorCfg(
        name="g", link_name="pelvis", measure="ang_vel", bias_range=(-0.1, 0.1)
    ).build()
    sensor.bind(env)
    first = sensor.data.clone()
    assert (first.abs() <= 0.1 + 1e-6).all()
    assert first.std() > 0.0  # initial bias should already vary across envs
    sensor.reset(torch.arange(64))
    second = sensor.data
    assert not torch.equal(first, second)


def test_body_velocity_sensor_rejects_unknown_link() -> None:
    env = _FakeRobotEnv()
    sensor = BodyVelocitySensorCfg(name="x", link_name="nonexistent").build()
    try:
        sensor.bind(env)
    except ValueError as exc:
        assert "nonexistent" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown link_name")


# --------------------------------------------------------------------- ContactSensor


class _FakeRobot:
    def __init__(self, contact_force: torch.Tensor) -> None:
        # contact_force shape: (num_envs, num_links, 3)
        self._contact_force = contact_force

    def get_links_net_contact_force(self) -> torch.Tensor:
        return self._contact_force

    def set_contact_force(self, contact_force: torch.Tensor) -> None:
        self._contact_force = contact_force


class _FakeContactEnv:
    def __init__(self, num_envs: int, link_names: tuple[str, ...]) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.link_names = list(link_names)
        self.robot = _FakeRobot(torch.zeros(num_envs, len(link_names), 3))


def test_contact_sensor_explicit_link_names_resolves_indices() -> None:
    env = _FakeContactEnv(num_envs=2, link_names=("base", "left_foot", "right_foot", "head"))
    sensor = ContactSensorCfg(
        name="feet", link_names=("left_foot", "right_foot"), track_air_time=False
    ).build()
    sensor.bind(env)
    assert sensor.link_names == ["left_foot", "right_foot"]
    env.robot.set_contact_force(
        torch.tensor(
            [
                [[0, 0, 0], [0, 0, 50.0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 30.0], [0, 0, 0]],
            ]
        )
    )
    sensor.update(0.02)
    data = sensor.data
    assert data.force_norm.shape == (2, 2)
    assert torch.allclose(data.force_norm[0], torch.tensor([50.0, 0.0]))
    assert torch.allclose(data.force_norm[1], torch.tensor([0.0, 30.0]))
    assert data.found.dtype == torch.bool
    assert torch.equal(data.found, torch.tensor([[True, False], [False, True]]))


def test_contact_sensor_regex_match() -> None:
    env = _FakeContactEnv(num_envs=1, link_names=("base", "left_foot", "right_foot"))
    sensor = ContactSensorCfg(name="f", link_names_expr=r"_foot$", track_air_time=False).build()
    sensor.bind(env)
    assert sensor.link_names == ["left_foot", "right_foot"]


def test_contact_sensor_air_time_state_machine() -> None:
    env = _FakeContactEnv(num_envs=1, link_names=("foot",))
    sensor = ContactSensorCfg(name="c", link_names=("foot",), track_air_time=True).build()
    sensor.bind(env)

    in_air = torch.zeros(1, 1, 3)
    in_contact = torch.tensor([[[0.0, 0.0, 100.0]]])
    dt = 0.05

    env.robot.set_contact_force(in_air)
    sensor.update(dt)  # tick 1 in air
    assert torch.allclose(sensor.data.current_air_time, torch.tensor([[dt]]))
    assert torch.allclose(sensor.data.current_contact_time, torch.tensor([[0.0]]))

    sensor._invalidate_cache()
    sensor.update(dt)  # tick 2 in air
    assert torch.allclose(sensor.data.current_air_time, torch.tensor([[2 * dt]]))

    env.robot.set_contact_force(in_contact)
    sensor._invalidate_cache()
    sensor.update(dt)  # landing tick
    d = sensor.data
    assert torch.allclose(d.current_air_time, torch.tensor([[0.0]]))
    assert torch.allclose(d.last_air_time, torch.tensor([[3 * dt]]))
    assert torch.allclose(d.current_contact_time, torch.tensor([[dt]]))

    sensor._invalidate_cache()
    sensor.update(dt)  # continued contact
    assert torch.allclose(sensor.data.current_contact_time, torch.tensor([[2 * dt]]))

    env.robot.set_contact_force(in_air)
    sensor._invalidate_cache()
    sensor.update(dt)  # lift-off
    d = sensor.data
    assert torch.allclose(d.current_contact_time, torch.tensor([[0.0]]))
    assert torch.allclose(d.last_contact_time, torch.tensor([[3 * dt]]))
    assert torch.allclose(d.current_air_time, torch.tensor([[dt]]))


def test_contact_sensor_reset_clears_state_for_env_ids_only() -> None:
    env = _FakeContactEnv(num_envs=2, link_names=("foot",))
    sensor = ContactSensorCfg(name="c", link_names=("foot",), track_air_time=True).build()
    sensor.bind(env)
    env.robot.set_contact_force(torch.zeros(2, 1, 3))
    for _ in range(3):
        sensor._invalidate_cache()
        sensor.update(0.1)
    assert torch.allclose(sensor.data.current_air_time, torch.tensor([[0.3], [0.3]]))
    sensor.reset(torch.tensor([0]))
    sensor._invalidate_cache()
    sensor.update(0.1)
    out = sensor.data.current_air_time
    assert torch.allclose(out[0], torch.tensor([0.1]))
    assert torch.allclose(out[1], torch.tensor([0.4]))


def test_contact_sensor_force_threshold_controls_found_bit() -> None:
    env = _FakeContactEnv(num_envs=1, link_names=("foot",))
    sensor = ContactSensorCfg(
        name="c", link_names=("foot",), force_threshold=10.0, track_air_time=False
    ).build()
    sensor.bind(env)
    env.robot.set_contact_force(torch.tensor([[[0.0, 0.0, 5.0]]]))
    sensor.update(0.02)
    assert sensor.data.found.item() is False
    env.robot.set_contact_force(torch.tensor([[[0.0, 0.0, 12.0]]]))
    sensor._invalidate_cache()
    sensor.update(0.02)
    assert sensor.data.found.item() is True


def test_contact_sensor_rejects_unresolved_links() -> None:
    env = _FakeContactEnv(num_envs=1, link_names=("a", "b"))
    try:
        ContactSensorCfg(name="x", link_names=("c",)).build().bind(env)
    except ValueError as exc:
        assert "'c'" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown link")


def test_contact_sensor_first_contact_and_first_detached_mark_transitions() -> None:
    """``first_contact`` fires on air→contact ticks only; ``first_detached`` on contact→air."""
    env = _FakeContactEnv(num_envs=1, link_names=("foot",))
    sensor = ContactSensorCfg(name="c", link_names=("foot",), track_air_time=True).build()
    sensor.bind(env)

    in_air = torch.zeros(1, 1, 3)
    in_contact = torch.tensor([[[0.0, 0.0, 100.0]]])
    dt = 0.05

    # Two air ticks: neither edge fires (the very first contact would need a prior air phase
    # but ``current_air_time`` starts at 0, so the transition predicate guards correctly).
    env.robot.set_contact_force(in_air)
    sensor.update(dt)
    assert sensor.data.first_contact.item() is False
    assert sensor.data.first_detached.item() is False

    sensor._invalidate_cache()
    sensor.update(dt)
    assert sensor.data.first_contact.item() is False

    # Landing tick: contact + ``current_air_time > 0`` → ``first_contact`` fires once.
    env.robot.set_contact_force(in_contact)
    sensor._invalidate_cache()
    sensor.update(dt)
    assert sensor.data.first_contact.item() is True
    assert sensor.data.first_detached.item() is False

    # Continued contact: edge no longer fires (air_time is 0).
    sensor._invalidate_cache()
    sensor.update(dt)
    assert sensor.data.first_contact.item() is False

    # Lift-off tick: air + ``current_contact_time > 0`` → ``first_detached`` fires once.
    env.robot.set_contact_force(in_air)
    sensor._invalidate_cache()
    sensor.update(dt)
    assert sensor.data.first_detached.item() is True
    assert sensor.data.first_contact.item() is False


def test_contact_sensor_first_contact_clears_on_reset() -> None:
    env = _FakeContactEnv(num_envs=2, link_names=("foot",))
    sensor = ContactSensorCfg(name="c", link_names=("foot",), track_air_time=True).build()
    sensor.bind(env)
    # Force a first_contact tick in both envs.
    env.robot.set_contact_force(torch.zeros(2, 1, 3))
    sensor.update(0.05)
    env.robot.set_contact_force(torch.tensor([[[0, 0, 100.0]], [[0, 0, 100.0]]]))
    sensor._invalidate_cache()
    sensor.update(0.05)
    assert torch.equal(sensor.data.first_contact, torch.tensor([[True], [True]]))
    # Resetting env 0 should clear its first_contact while env 1's stays set until next tick.
    sensor.reset(torch.tensor([0]))
    sensor._invalidate_cache()
    # next update will overwrite both anyway, but pre-update state must reflect reset.
    assert sensor._air_state is not None
    assert torch.equal(sensor._air_state.first_contact, torch.tensor([[False], [True]]))


def test_contact_sensor_force_history_buffer_writes_and_wraps() -> None:
    """``history_length=3`` keeps the most recent 3 force snapshots; the buffer is
    written in place at an advancing head pointer, so after >H steps it wraps."""
    env = _FakeContactEnv(num_envs=1, link_names=("foot",))
    sensor = ContactSensorCfg(
        name="c", link_names=("foot",), track_air_time=False, history_length=3
    ).build()
    sensor.bind(env)
    assert sensor._force_history is not None
    assert sensor._force_history.shape == (1, 1, 3, 3)

    forces = [10.0, 20.0, 30.0, 40.0]
    for i, fz in enumerate(forces):
        env.robot.set_contact_force(torch.tensor([[[0.0, 0.0, fz]]]))
        sensor._invalidate_cache()
        sensor.update(0.05)
        # After step i (i ∈ 0..3), head has advanced (i+1) % 3 times.
        assert sensor._history_head == (i + 1) % 3

    # After 4 writes into a 3-slot ring, the buffer holds the latest three magnitudes
    # in write-pointer order. We don't care about order — just the set.
    z_mags = sensor._force_history[0, 0, :, 2].tolist()
    assert sorted(z_mags) == [20.0, 30.0, 40.0]


def test_contact_sensor_force_history_exposed_via_data() -> None:
    env = _FakeContactEnv(num_envs=2, link_names=("foot",))
    sensor = ContactSensorCfg(
        name="c", link_names=("foot",), track_air_time=False, history_length=4
    ).build()
    sensor.bind(env)
    env.robot.set_contact_force(torch.tensor([[[0, 0, 5.0]], [[0, 0, 50.0]]]))
    sensor.update(0.05)
    fh = sensor.data.force_history
    assert fh is not None
    assert fh.shape == (2, 1, 4, 3)


def test_contact_sensor_force_history_none_by_default() -> None:
    env = _FakeContactEnv(num_envs=1, link_names=("foot",))
    sensor = ContactSensorCfg(name="c", link_names=("foot",), track_air_time=True).build()
    sensor.bind(env)
    env.robot.set_contact_force(torch.tensor([[[0, 0, 100.0]]]))
    sensor.update(0.05)
    assert sensor.data.force_history is None


def test_contact_sensor_force_history_reset_clears_env_ids_only() -> None:
    env = _FakeContactEnv(num_envs=2, link_names=("foot",))
    sensor = ContactSensorCfg(
        name="c", link_names=("foot",), track_air_time=False, history_length=2
    ).build()
    sensor.bind(env)
    env.robot.set_contact_force(torch.tensor([[[0, 0, 30.0]], [[0, 0, 30.0]]]))
    sensor.update(0.05)
    sensor._invalidate_cache()
    sensor.update(0.05)
    sensor.reset(torch.tensor([0]))
    assert sensor._force_history is not None
    assert torch.all(sensor._force_history[0] == 0.0)
    assert torch.any(sensor._force_history[1] != 0.0)


# --------------------------------------------------------------------- RayCast / TerrainHeight


class _FakeTerrainState:
    def __init__(
        self,
        num_envs: int,
        num_links: int,
        device: str,
        link_pos: torch.Tensor,
    ) -> None:
        self.link_pos = link_pos
        self.link_quat_w = torch.zeros(num_envs, num_links, 4, device=device)
        self.link_quat_w[..., 0] = 1.0
        self.link_lin_vel_w = torch.zeros(num_envs, num_links, 3, device=device)
        self.link_ang_vel_w = torch.zeros(num_envs, num_links, 3, device=device)


class _FakeTerrainEnv:
    def __init__(
        self,
        num_envs: int,
        link_names: tuple[str, ...],
        link_pos: torch.Tensor,
    ) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.link_names = list(link_names)
        self.robot_state = _FakeTerrainState(num_envs, len(link_names), self.device, link_pos)


def test_grid_pattern_generates_expected_count_and_shape() -> None:
    pattern = GridPattern(resolution=0.5, size=(2.0, 1.0), direction=(0.0, 0.0, -1.0))
    starts, dirs = pattern.generate("cpu")
    assert pattern.num_rays() == 5 * 3
    assert starts.shape == (15, 3)
    assert dirs.shape == (15, 3)
    assert torch.allclose(dirs[0], torch.tensor([0.0, 0.0, -1.0]))
    # Outer corners are at the configured extent.
    assert torch.allclose(starts[..., 0].max(), torch.tensor(1.0))
    assert torch.allclose(starts[..., 0].min(), torch.tensor(-1.0))


def test_raycast_sensor_returns_link_z_distance_on_flat_plane() -> None:
    link_pos = torch.tensor([[[0.0, 0.0, 1.2]]])
    env = _FakeTerrainEnv(1, ("torso",), link_pos)
    sensor = RayCastSensorCfg(
        name="r",
        link_name="torso",
        pattern=GridPattern(resolution=0.5, size=(1.0, 1.0)),
        max_distance=5.0,
    ).build()
    sensor.bind(env)
    data = sensor.data
    # 9 rays in a 3x3 grid; each starts at link_pos.z=1.2 and hits z=0 → distance 1.2.
    assert data.distances.shape == (1, 9)
    assert torch.allclose(data.distances, torch.full((1, 9), 1.2))
    assert torch.allclose(data.hit_pos_w[..., 2], torch.zeros(1, 9))
    assert torch.allclose(data.normals_w[..., 2], torch.ones(1, 9))


def test_raycast_sensor_clamps_at_max_distance_when_no_intersection() -> None:
    # Link is below ground → upward-only hit; default backend treats as miss → max_distance.
    link_pos = torch.tensor([[[0.0, 0.0, -1.0]]])
    env = _FakeTerrainEnv(1, ("torso",), link_pos)
    sensor = RayCastSensorCfg(
        name="r",
        link_name="torso",
        pattern=GridPattern(resolution=1.0, size=(0.0, 0.0)),
        max_distance=3.0,
    ).build()
    sensor.bind(env)
    data = sensor.data
    assert torch.allclose(data.distances, torch.full_like(data.distances, 3.0))


def test_terrain_height_sensor_returns_link_height_on_flat_plane() -> None:
    # Two envs at different heights — height_scan output is constant across rays.
    link_pos = torch.tensor([[[0.0, 0.0, 0.8]], [[1.0, 1.0, 1.5]]])
    env = _FakeTerrainEnv(2, ("torso",), link_pos)
    sensor = TerrainHeightSensorCfg(
        name="h",
        link_name="torso",
        pattern=GridPattern(resolution=0.5, size=(1.0, 1.0)),
    ).build()
    sensor.bind(env)
    heights = sensor.data
    assert heights.shape == (2, 9)
    assert torch.allclose(heights[0], torch.full((9,), 0.8))
    assert torch.allclose(heights[1], torch.full((9,), 1.5))


def test_terrain_height_sensor_lifecycle_propagates_to_inner_sensor() -> None:
    link_pos = torch.tensor([[[0.0, 0.0, 1.0]]])
    env = _FakeTerrainEnv(1, ("torso",), link_pos)
    sensor = TerrainHeightSensorCfg(
        name="h",
        link_name="torso",
        pattern=GridPattern(resolution=1.0, size=(0.0, 0.0)),
    ).build()
    sensor.bind(env)
    _ = sensor.data
    assert sensor._cache_valid is True
    sensor.update(0.02)
    assert sensor._cache_valid is False
    _ = sensor.data
    sensor.reset(torch.tensor([0]))
    assert sensor._cache_valid is False


def test_terrain_height_sensor_multi_frame_min_reduction_per_link() -> None:
    """Two-foot scan with ``reduction='min'`` returns ``(B, F=2)`` per-foot min heights."""
    # Left foot at z=0.2, right foot at z=0.5. All rays scan straight down to z=0.
    link_pos = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.5]]])
    env = _FakeTerrainEnv(1, ("base", "left_foot", "right_foot"), link_pos)
    sensor = TerrainHeightSensorCfg(
        name="feet_h",
        link_names=("left_foot", "right_foot"),
        pattern=GridPattern(resolution=0.5, size=(0.5, 0.5)),
        reduction="min",
    ).build()
    sensor.bind(env)
    heights = sensor.data
    assert heights.shape == (1, 2)
    assert torch.allclose(heights[0], torch.tensor([0.2, 0.5]), atol=1e-6)


def test_terrain_height_sensor_multi_frame_reduction_none_keeps_per_ray_axis() -> None:
    link_pos = torch.tensor([[[0.0, 0.0, 0.3], [0.0, 0.0, 0.7]]])
    env = _FakeTerrainEnv(1, ("left_foot", "right_foot"), link_pos)
    sensor = TerrainHeightSensorCfg(
        name="feet_h_raw",
        link_names=("left_foot", "right_foot"),
        pattern=GridPattern(resolution=0.5, size=(0.5, 0.5)),  # 2x2 = 4 rays
        reduction="none",
    ).build()
    sensor.bind(env)
    heights = sensor.data
    # 2 frames × 4 rays per frame → (B, F, num_rays).
    assert heights.shape == (1, 2, 4)
    assert torch.allclose(heights[0, 0], torch.full((4,), 0.3), atol=1e-6)
    assert torch.allclose(heights[0, 1], torch.full((4,), 0.7), atol=1e-6)


def test_terrain_height_sensor_rejects_both_link_name_modes() -> None:
    try:
        TerrainHeightSensorCfg(name="bad", link_name="torso", link_names=("foot",))
    except ValueError as exc:
        assert "link_name" in str(exc) and "link_names" in str(exc)
    else:
        raise AssertionError("expected ValueError when both link_name and link_names are set")


def test_terrain_height_sensor_rejects_neither_link_mode() -> None:
    try:
        TerrainHeightSensorCfg(name="empty")
    except ValueError as exc:
        assert "link_name" in str(exc)
    else:
        raise AssertionError("expected ValueError when neither link_name nor link_names set")


# --------------------------------------------------------------------- Pattern: Ring / Hemisphere


def test_ring_pattern_full_sweep_does_not_duplicate_endpoint() -> None:
    pattern = RingPattern(
        num_horizontal=8,
        num_vertical=1,
        horizontal_fov_deg=(-180.0, 180.0),
        vertical_fov_deg=(0.0, 0.0),
    )
    starts, dirs = pattern.generate("cpu")
    assert pattern.num_rays() == 8
    assert starts.shape == (8, 3) and dirs.shape == (8, 3)
    assert torch.allclose(starts, torch.zeros(8, 3))
    # All rays sit in the horizontal plane.
    assert torch.allclose(dirs[..., 2], torch.zeros(8), atol=1e-6)
    # Unit norm.
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(8), atol=1e-6)
    # 360° sweep with N=8 means azimuth step = 45°; first ray points along -x.
    # The closing endpoint (+180°) must not equal the opening (-180°): both map to (-1, 0, 0)
    # in cartesian. Confirm no duplicate by checking the last ray is at +135°, not +180°.
    last_dir = dirs[-1]
    expected_last = torch.tensor(
        [math.cos(math.radians(135.0)), math.sin(math.radians(135.0)), 0.0]
    )
    assert torch.allclose(last_dir, expected_last, atol=1e-6)


def test_ring_pattern_multi_line_lidar_layout() -> None:
    pattern = RingPattern(
        num_horizontal=4,
        num_vertical=3,
        horizontal_fov_deg=(0.0, 90.0),
        vertical_fov_deg=(-10.0, 10.0),
    )
    starts, dirs = pattern.generate("cpu")
    assert pattern.num_rays() == 12
    assert dirs.shape == (12, 3)
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(12), atol=1e-6)
    # Vertical varies slowest (outer index): first 4 rays share the same elevation.
    assert torch.allclose(dirs[0, 2], dirs[3, 2], atol=1e-6)
    assert not torch.isclose(dirs[0, 2], dirs[4, 2], atol=1e-6)


def test_hemisphere_pattern_covers_pole_axis_half_space() -> None:
    pattern = HemispherePattern(num_rays_target=128, pole_axis=(0.0, 0.0, 1.0), polar_fov_deg=90.0)
    starts, dirs = pattern.generate("cpu")
    assert pattern.num_rays() == 128
    assert dirs.shape == (128, 3)
    # All directions are unit length.
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(128), atol=1e-5)
    # On a hemisphere around +z all rays have dot(d, +z) >= 0.
    pole = torch.tensor([0.0, 0.0, 1.0])
    assert (dirs @ pole >= -1e-6).all()
    # Mean direction tracks the pole (Fibonacci lattice is well-distributed).
    assert torch.allclose(dirs.mean(dim=0)[2], torch.tensor(0.5), atol=0.1)


def test_hemisphere_pattern_rotates_to_arbitrary_pole_axis() -> None:
    pattern = HemispherePattern(num_rays_target=64, pole_axis=(1.0, 0.0, 0.0), polar_fov_deg=90.0)
    _, dirs = pattern.generate("cpu")
    pole = torch.tensor([1.0, 0.0, 0.0])
    assert (dirs @ pole >= -1e-6).all()
    # No ray points opposite to the pole.
    assert (dirs @ pole > -0.01).all()


def test_hemisphere_pattern_antipodal_pole_axis_flips_correctly() -> None:
    pattern = HemispherePattern(num_rays_target=32, pole_axis=(0.0, 0.0, -1.0), polar_fov_deg=90.0)
    _, dirs = pattern.generate("cpu")
    # All rays point downward (z <= 0) since the pole is -z.
    assert (dirs[..., 2] <= 1e-6).all()
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(32), atol=1e-5)


def test_raycast_sensor_accepts_ring_pattern_against_flat_plane() -> None:
    # Sensor at z=1.0 above the flat plane, all rays tilted 30° below horizon.
    link_pos = torch.tensor([[[0.0, 0.0, 1.0]]])
    env = _FakeTerrainEnv(1, ("torso",), link_pos)
    sensor = RayCastSensorCfg(
        name="lidar",
        link_name="torso",
        pattern=RingPattern(
            num_horizontal=4,
            num_vertical=1,
            horizontal_fov_deg=(-180.0, 180.0),
            vertical_fov_deg=(-30.0, -30.0),
        ),
        max_distance=10.0,
    ).build()
    sensor.bind(env)
    data = sensor.data
    assert data.distances.shape == (1, 4)
    # Distance along a 30°-down ray to hit z=0 from z=1.0 is 1.0 / sin(30°) = 2.0.
    assert torch.allclose(data.distances, torch.full((1, 4), 2.0), atol=1e-5)
    # All hit points sit on the plane z=0.
    assert torch.allclose(data.hit_pos_w[..., 2], torch.zeros(1, 4), atol=1e-5)


def test_raycast_sensor_with_upward_hemisphere_clamps_to_max_distance() -> None:
    # Hemisphere pointing +z: no ray hits the flat plane below.
    link_pos = torch.tensor([[[0.0, 0.0, 0.5]]])
    env = _FakeTerrainEnv(1, ("torso",), link_pos)
    sensor = RayCastSensorCfg(
        name="dome",
        link_name="torso",
        pattern=HemispherePattern(
            num_rays_target=16, pole_axis=(0.0, 0.0, 1.0), polar_fov_deg=80.0
        ),
        max_distance=4.0,
    ).build()
    sensor.bind(env)
    data = sensor.data
    assert data.distances.shape == (1, 16)
    assert torch.allclose(data.distances, torch.full((1, 16), 4.0))


# --------------------------------------------------------------------- IMUSensor


def test_imu_sensor_static_link_reads_negative_body_gravity() -> None:
    env = _FakeRobotEnv()
    sensor = IMUSensorCfg(name="imu", link_name="pelvis", gravity_bias=True).build()
    sensor.bind(env)
    sensor.update(0.02)
    data = sensor.data
    assert data.orientation.shape == (2, 4)
    assert torch.allclose(data.orientation[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    # Identity quat → projected gravity is (0, 0, -1).
    assert torch.allclose(data.projected_gravity_b[0], torch.tensor([0.0, 0.0, -1.0]))
    # First step zero-acceleration + gravity bias → lin_acc_b ≈ -g_b = (0, 0, +9.81).
    assert torch.allclose(data.lin_acc_b[0], torch.tensor([0.0, 0.0, 9.81]), atol=1e-5)
    assert torch.allclose(data.ang_acc_b[0], torch.zeros(3), atol=1e-6)


def test_imu_sensor_first_step_after_reset_returns_zero_acc() -> None:
    env = _FakeRobotEnv()
    sensor = IMUSensorCfg(name="imu", link_name="pelvis", gravity_bias=False).build()
    sensor.bind(env)
    # Prime the sensor with a non-trivial velocity to seed _prev.
    env.robot_state.link_lin_vel_w[:, 0] = torch.tensor([1.0, 0.0, 0.0])
    env.robot_state.link_ang_vel_w[:, 0] = torch.tensor([0.0, 1.0, 0.0])
    sensor.update(0.02)
    # Now reset env 0 → first-step flag re-armed.
    sensor.reset(torch.tensor([0]))
    env.robot_state.link_lin_vel_w[:, 0] = torch.tensor([5.0, 0.0, 0.0])
    sensor.update(0.02)
    data = sensor.data
    assert torch.allclose(data.lin_acc_b[0], torch.zeros(3), atol=1e-6)
    assert torch.allclose(data.ang_acc_b[0], torch.zeros(3), atol=1e-6)


def test_imu_sensor_finite_diff_linear_acceleration() -> None:
    # Two ticks with constant world-frame velocity ramp v = a * t → lin_acc_b ≈ a.
    env = _FakeRobotEnv()
    sensor = IMUSensorCfg(name="imu", link_name="pelvis", gravity_bias=False).build()
    sensor.bind(env)
    dt = 0.02
    a = torch.tensor([2.0, -1.5, 0.5])
    env.robot_state.link_lin_vel_w[:, 0] = 0.0 * a
    sensor.update(dt)  # primes prev
    env.robot_state.link_lin_vel_w[:, 0] = 1.0 * dt * a
    sensor.update(dt)
    data = sensor.data
    expected = a
    assert torch.allclose(data.lin_acc_b[0], expected, atol=1e-5)


def test_imu_sensor_finite_diff_angular_acceleration() -> None:
    env = _FakeRobotEnv()
    sensor = IMUSensorCfg(name="imu", link_name="pelvis", gravity_bias=False).build()
    sensor.bind(env)
    dt = 0.02
    env.robot_state.link_ang_vel_w[:, 0] = torch.tensor([1.0, 0.0, 0.0])
    sensor.update(dt)  # primes prev
    env.robot_state.link_ang_vel_w[:, 0] = torch.tensor([1.1, 0.0, 0.0])
    sensor.update(dt)
    data = sensor.data
    assert torch.allclose(data.ang_acc_b[0], torch.tensor([0.1 / dt, 0.0, 0.0]), atol=1e-5)


def test_imu_sensor_lin_acc_bias_randomizes_on_reset() -> None:
    torch.manual_seed(0)
    env = _FakeRobotEnv(num_envs=64)
    sensor = IMUSensorCfg(
        name="imu",
        link_name="pelvis",
        gravity_bias=False,
        bias_range_lin_acc=(-0.5, 0.5),
    ).build()
    sensor.bind(env)
    sensor.update(0.02)
    first = sensor.data.lin_acc_b.clone()
    # First-step acceleration is zero across envs; bias is the sole non-zero contribution.
    assert (first.abs() <= 0.5 + 1e-6).all()
    assert first.std() > 0.0
    sensor.reset(torch.arange(64))
    sensor.update(0.02)
    second = sensor.data.lin_acc_b
    assert not torch.equal(first, second)


def test_imu_sensor_coexists_with_body_velocity_sensor() -> None:
    env = _FakeRobotEnv()
    env.robot_state.link_lin_vel_w[:, 0] = torch.tensor([1.0, 0.0, 0.0])
    env.robot_state.link_ang_vel_w[:, 0] = torch.tensor([0.0, 0.0, 0.5])
    imu = IMUSensorCfg(name="imu", link_name="pelvis", gravity_bias=False).build()
    bv = BodyVelocitySensorCfg(name="bv", link_name="pelvis", measure="lin_vel").build()
    imu.bind(env)
    bv.bind(env)
    imu.update(0.02)
    bv.update(0.02)
    # IMU at first step returns zero accel; BodyVelocity returns body-frame velocity.
    imu_data = imu.data
    bv_data = bv.data
    assert torch.allclose(imu_data.lin_acc_b[0], torch.zeros(3), atol=1e-6)
    assert torch.allclose(bv_data[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


def test_imu_sensor_rejects_unknown_link() -> None:
    env = _FakeRobotEnv()
    sensor = IMUSensorCfg(name="x", link_name="nonexistent").build()
    try:
        sensor.bind(env)
    except ValueError as exc:
        assert "nonexistent" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown link_name")


# --------------------------------------------------------------------- FrameTransformerSensor


def test_frame_transformer_single_target_zero_offset_matches_world_pose() -> None:
    link_pos = torch.tensor([[[0.0, 0.0, 0.0], [1.5, -0.3, 0.7]]])
    env = _FakeTerrainEnv(1, ("base", "ee"), link_pos)
    sensor = FrameTransformerSensorCfg(
        name="ee",
        source_link_name="base",
        target_frames=(TargetFrameCfg(link_name="ee"),),
    ).build()
    sensor.bind(env)
    data = sensor.data
    assert data.target_pos_w.shape == (1, 1, 3)
    assert data.target_quat_w.shape == (1, 1, 4)
    # Source at world origin with identity quat → target_pos_source equals target_pos_w.
    assert torch.allclose(data.target_pos_source[0, 0], torch.tensor([1.5, -0.3, 0.7]), atol=1e-6)
    assert torch.allclose(data.target_quat_source[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_frame_transformer_target_offset_applies_in_link_frame() -> None:
    link_pos = torch.tensor([[[2.0, 0.0, 0.0]]])
    env = _FakeTerrainEnv(1, ("wrist",), link_pos)
    sensor = FrameTransformerSensorCfg(
        name="grip",
        source_link_name="wrist",
        target_frames=(TargetFrameCfg(link_name="wrist", offset_pos=(0.1, 0.0, 0.0)),),
    ).build()
    sensor.bind(env)
    data = sensor.data
    # Target = same link + 0.1 along +x in link frame → world (2.1, 0, 0); relative to
    # source (the same link) → (0.1, 0, 0).
    assert torch.allclose(data.target_pos_w[0, 0], torch.tensor([2.1, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(data.target_pos_source[0, 0], torch.tensor([0.1, 0.0, 0.0]), atol=1e-6)


def test_frame_transformer_source_offset_shifts_relative_pose() -> None:
    # Source link at world origin, source offset +x; target at world (0.5, 0, 0).
    # target_pos_source = (target_pos_w - source_pos_w) rotated by inv(identity)
    #                   = (0.5 - 1.0, 0, 0) = (-0.5, 0, 0).
    link_pos = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]])
    env = _FakeTerrainEnv(1, ("base", "target"), link_pos)
    sensor = FrameTransformerSensorCfg(
        name="rel",
        source_link_name="base",
        source_offset_pos=(1.0, 0.0, 0.0),
        target_frames=(TargetFrameCfg(link_name="target"),),
    ).build()
    sensor.bind(env)
    data = sensor.data
    assert torch.allclose(data.target_pos_source[0, 0], torch.tensor([-0.5, 0.0, 0.0]), atol=1e-6)


def test_frame_transformer_preserves_target_order() -> None:
    # Three targets in cfg order; expected N-axis order matches.
    link_pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]])
    env = _FakeTerrainEnv(1, ("base", "a", "b", "c"), link_pos)
    sensor = FrameTransformerSensorCfg(
        name="multi",
        source_link_name="base",
        target_frames=(
            TargetFrameCfg(link_name="c", name="z_axis"),
            TargetFrameCfg(link_name="a", name="x_axis"),
            TargetFrameCfg(link_name="b", name="y_axis"),
        ),
    ).build()
    sensor.bind(env)
    data = sensor.data
    assert sensor.target_names == ("z_axis", "x_axis", "y_axis")
    assert torch.allclose(data.target_pos_w[0, 0], torch.tensor([0.0, 0.0, 3.0]), atol=1e-6)
    assert torch.allclose(data.target_pos_w[0, 1], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(data.target_pos_w[0, 2], torch.tensor([0.0, 2.0, 0.0]), atol=1e-6)


def test_frame_transformer_rejects_unknown_link() -> None:
    link_pos = torch.tensor([[[0.0, 0.0, 0.0]]])
    env = _FakeTerrainEnv(1, ("base",), link_pos)
    sensor = FrameTransformerSensorCfg(
        name="bad",
        source_link_name="base",
        target_frames=(
            TargetFrameCfg(link_name="missing_a"),
            TargetFrameCfg(link_name="missing_b"),
        ),
    ).build()
    try:
        sensor.bind(env)
    except ValueError as exc:
        msg = str(exc)
        assert "missing_a" in msg
        assert "missing_b" in msg
    else:
        raise AssertionError("expected ValueError for unknown target link(s)")


def test_frame_transformer_rejects_empty_target_frames() -> None:
    link_pos = torch.tensor([[[0.0, 0.0, 0.0]]])
    env = _FakeTerrainEnv(1, ("base",), link_pos)
    sensor = FrameTransformerSensorCfg(name="empty", source_link_name="base").build()
    try:
        sensor.bind(env)
    except ValueError as exc:
        assert "target" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for empty target_frames")


# ---------------------------------------------------------------------------- camera


class _FakeGsLink:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeGsRobot:
    def __init__(self, link_names: tuple[str, ...]) -> None:
        self._links = {name: _FakeGsLink(name) for name in link_names}

    def get_link(self, name: str) -> _FakeGsLink:
        return self._links[name]


class _FakeGsCamera:
    def __init__(self, *, num_envs: int, **kwargs: Any) -> None:
        self.kwargs: dict[str, Any] = kwargs
        self.num_envs = num_envs
        self.attached_link: _FakeGsLink | None = None
        self.attached_offset_T: Any = None
        self.move_calls: int = 0
        self.render_calls: int = 0

    def attach(self, link: Any, offset_T: Any) -> None:
        self.attached_link = link
        self.attached_offset_T = offset_T

    def move_to_attach(self) -> None:
        self.move_calls += 1

    def render(
        self,
        rgb: bool = False,
        depth: bool = False,
        segmentation: bool = False,
        colorize_seg: bool = False,
        normal: bool = False,  # noqa: ARG002
    ) -> tuple[Any, Any, Any, Any]:
        self.render_calls += 1
        h = int(self.kwargs["res"][1])
        w = int(self.kwargs["res"][0])
        rgb_t = torch.zeros(self.num_envs, h, w, 3, dtype=torch.uint8) if rgb else None
        depth_t = torch.full((self.num_envs, h, w), 0.5) if depth else None
        seg_t = None
        if segmentation:
            seg_t = (
                torch.zeros(self.num_envs, h, w, 3, dtype=torch.uint8)
                if colorize_seg
                else torch.ones(self.num_envs, h, w, dtype=torch.int32)
            )
        return rgb_t, depth_t, seg_t, None


class _FakeGsScene:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.cameras: list[_FakeGsCamera] = []

    def add_camera(self, **kwargs: Any) -> _FakeGsCamera:
        cam = _FakeGsCamera(num_envs=self.num_envs, **kwargs)
        self.cameras.append(cam)
        return cam


class _FakeInteractiveScene:
    def __init__(self, num_envs: int) -> None:
        self.gs_scene = _FakeGsScene(num_envs)


class _FakeCameraEnv:
    """Extends ``_FakeEnv`` with the surface the camera sensor reaches into."""

    def __init__(self, num_envs: int = 2, link_names: tuple[str, ...] = ("base", "head")) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.sensors: dict[str, Sensor[Any]] = {}
        self.link_names: list[str] = list(link_names)
        self.robot = _FakeGsRobot(link_names)
        self.scene = _FakeInteractiveScene(num_envs)


def test_camera_bind_attaches_to_link() -> None:
    env = _FakeCameraEnv(num_envs=2, link_names=("base", "head"))
    sensor = CameraSensorCfg(
        name="cam",
        link_name="head",
        offset_pos=(0.05, 0.0, 0.15),
        offset_quat=(1.0, 0.0, 0.0, 0.0),
        width=4,
        height=4,
    ).build()
    sensor.bind(env)  # type: ignore[arg-type]
    cam = env.scene.gs_scene.cameras[0]
    assert isinstance(cam.attached_link, _FakeGsLink)
    assert cam.attached_link.name == "head"
    assert cam.attached_offset_T.shape == (4, 4)
    # Identity rotation → upper-left 3×3 is the identity.
    assert np.allclose(cam.attached_offset_T[:3, :3], np.eye(3))
    # Translation block reflects ``offset_pos``.
    assert np.allclose(cam.attached_offset_T[:3, 3], np.array([0.05, 0.0, 0.15]))
    assert cam.attached_offset_T[3, 3] == 1.0


def test_camera_compute_returns_rgb_and_depth() -> None:
    env = _FakeCameraEnv(num_envs=2)
    sensor = CameraSensorCfg(name="cam", link_name="head", width=4, height=4).build()
    sensor.bind(env)  # type: ignore[arg-type]
    cam = env.scene.gs_scene.cameras[0]
    data = sensor.data
    assert data.rgb is not None and data.depth is not None
    assert data.rgb.shape == (2, 4, 4, 3)
    assert data.depth.shape == (2, 4, 4)
    assert cam.move_calls == 1
    assert cam.render_calls == 1
    # Cache invalidates on update(), so the second access triggers another render.
    sensor.update(0.02)
    _ = sensor.data
    assert cam.move_calls == 2
    assert cam.render_calls == 2


def test_camera_render_rgb_only() -> None:
    env = _FakeCameraEnv(num_envs=2)
    sensor = CameraSensorCfg(
        name="cam", link_name="head", width=4, height=4, render_depth=False
    ).build()
    sensor.bind(env)  # type: ignore[arg-type]
    data = sensor.data
    assert data.rgb is not None
    assert data.depth is None


def test_camera_segmentation_index_map() -> None:
    env = _FakeCameraEnv(num_envs=2)
    sensor = CameraSensorCfg(
        name="cam", link_name="head", width=4, height=4, render_segmentation=True
    ).build()
    sensor.bind(env)  # type: ignore[arg-type]
    data = sensor.data
    assert data.segmentation is not None
    assert data.segmentation.shape == (2, 4, 4)
    assert data.segmentation.dtype == torch.int32


def test_camera_segmentation_colorized() -> None:
    env = _FakeCameraEnv(num_envs=2)
    sensor = CameraSensorCfg(
        name="cam",
        link_name="head",
        width=4,
        height=4,
        render_segmentation=True,
        colorize_segmentation=True,
    ).build()
    sensor.bind(env)  # type: ignore[arg-type]
    data = sensor.data
    assert data.segmentation is not None
    assert data.segmentation.shape == (2, 4, 4, 3)
    assert data.segmentation.dtype == torch.uint8


def test_camera_segmentation_off_by_default() -> None:
    env = _FakeCameraEnv(num_envs=2)
    sensor = CameraSensorCfg(name="cam", link_name="head", width=4, height=4).build()
    sensor.bind(env)  # type: ignore[arg-type]
    assert sensor.data.segmentation is None


def test_camera_segmentation_only_config_valid() -> None:
    env = _FakeCameraEnv(num_envs=2)
    sensor = CameraSensorCfg(
        name="cam",
        link_name="head",
        width=4,
        height=4,
        render_rgb=False,
        render_depth=False,
        render_segmentation=True,
    ).build()
    sensor.bind(env)  # type: ignore[arg-type]
    data = sensor.data
    assert data.rgb is None and data.depth is None
    assert data.segmentation is not None


def test_camera_unknown_link_raises() -> None:
    env = _FakeCameraEnv(num_envs=2, link_names=("base", "torso"))
    sensor = CameraSensorCfg(name="cam", link_name="head").build()
    try:
        sensor.bind(env)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "head" in str(exc)
        assert "env.link_names" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown link_name")


# --------------------------------------------------------------------- SelfContactSensor


class _FakeContactDictRobot:
    """Fake robot returning a controlled ``get_contacts(with_entity=…)`` dict."""

    def __init__(self, contacts: dict | None) -> None:
        self._contacts = contacts

    def get_contacts(self, with_entity=None) -> dict | None:  # noqa: ARG002 - param for signature parity
        return self._contacts


class _FakeSelfContactEnv:
    def __init__(self, num_envs: int, contacts: dict | None) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.robot = _FakeContactDictRobot(contacts)


def test_self_contact_sensor_zero_when_no_genesis_runtime() -> None:
    """Robot without ``get_contacts`` (typical for unit-test fakes) → all zeros."""

    class _NoContactRobot:
        pass

    env = _FakeSelfContactEnv(num_envs=2, contacts=None)
    env.robot = _NoContactRobot()  # type: ignore[assignment]
    sensor = SelfContactSensorCfg(name="self", force_threshold=1.0).build()
    sensor.bind(env)
    sensor.update(0.02)
    data = sensor.data
    assert torch.all(data.force == 0.0)
    assert not data.found.any().item()
    assert data.force_history is None


def test_self_contact_sensor_reports_max_pair_force_per_env() -> None:
    """mjlab parity: per-env signal is the *max* pair magnitude, not the sum."""
    contacts = {
        "force_a": torch.tensor(
            [
                [[3.0, 4.0, 0.0], [6.0, 8.0, 0.0]],  # env 0: |F|=5, |F|=10 → max=10
                [[3.0, 4.0, 0.0], [9.0, 0.0, 0.0]],  # env 1: max=5 (second masked)
            ]
        ),
        "valid_mask": torch.tensor([[True, True], [True, False]]),
    }
    env = _FakeSelfContactEnv(num_envs=2, contacts=contacts)
    sensor = SelfContactSensorCfg(name="self").build()
    sensor.bind(env)
    sensor.update(0.02)
    data = sensor.data
    assert torch.allclose(data.force, torch.tensor([10.0, 5.0]), atol=1e-6)


def test_self_contact_sensor_found_bit_is_any_pair_above_threshold() -> None:
    """mjlab parity: ``found`` is True iff at least one *single* pair exceeded thr.

    The pre-parity behaviour thresholded the *sum* of all pair forces; with that
    rule two 0.6-magnitude pairs would falsely flag a 1.0-threshold sensor (sum=1.2).
    The any-pair rule keeps each pair independent: each must clear the bar alone.
    """
    contacts = {
        # Two pairs of magnitude 0.6 each — sum=1.2 but max=0.6.
        "force_a": torch.tensor([[[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]]]),
        "valid_mask": torch.tensor([[True, True]]),
    }
    env = _FakeSelfContactEnv(num_envs=1, contacts=contacts)
    sensor = SelfContactSensorCfg(name="self", force_threshold=1.0).build()
    sensor.bind(env)
    sensor.update(0.02)
    assert sensor.data.found.item() is False  # would be True under the sum rule
    # Lowering the threshold catches the per-pair 0.6.
    sensor = SelfContactSensorCfg(name="self2", force_threshold=0.5).build()
    sensor.bind(env)
    sensor.update(0.02)
    assert sensor.data.found.item() is True


def test_self_contact_sensor_history_buffer_stores_any_above_bools() -> None:
    contacts = {
        "force_a": torch.tensor([[[2.0, 0.0, 0.0]]]),
        "valid_mask": torch.tensor([[True]]),
    }
    env = _FakeSelfContactEnv(num_envs=2, contacts=contacts)
    sensor = SelfContactSensorCfg(name="self", force_threshold=1.0, history_length=3).build()
    sensor.bind(env)
    for _ in range(3):
        sensor._invalidate_cache()
        sensor.update(0.02)
    fh = sensor.data.force_history
    assert fh is not None
    assert fh.shape == (2, 3)
    assert fh.dtype == torch.bool
    assert torch.all(fh)  # all three substeps tripped the threshold
    # Reset env 0 — its history should clear; env 1 stays untouched.
    sensor.reset(torch.tensor([0]))
    assert not sensor._force_history[0].any().item()  # type: ignore[index]
    assert sensor._force_history[1].all().item()  # type: ignore[index]


# --------------------------------------------------------------------- RootAngularMomentumSensor


class _FakeMassRobot:
    def __init__(self, masses: torch.Tensor | None) -> None:
        self._masses = masses

    def get_links_inertial_mass(self) -> torch.Tensor | None:
        return self._masses


class _FakeAngMomState:
    def __init__(
        self,
        root_pos: torch.Tensor,
        link_pos: torch.Tensor,
        link_lin_vel_w: torch.Tensor,
    ) -> None:
        self.root_pos = root_pos
        self.link_pos = link_pos
        self.link_lin_vel_w = link_lin_vel_w


class _FakeAngMomEnv:
    def __init__(
        self,
        num_envs: int,
        link_names: tuple[str, ...],
        masses: torch.Tensor | None,
        root_pos: torch.Tensor,
        link_pos: torch.Tensor,
        link_lin_vel_w: torch.Tensor,
    ) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.link_names = list(link_names)
        self.robot = _FakeMassRobot(masses)
        self.robot_state = _FakeAngMomState(root_pos, link_pos, link_lin_vel_w)


def test_root_angular_momentum_zero_when_links_static() -> None:
    """L = Σ m·r×v vanishes when every link velocity is zero."""
    env = _FakeAngMomEnv(
        num_envs=2,
        link_names=("base", "left_arm"),
        masses=torch.tensor([1.0, 0.5]),
        root_pos=torch.zeros(2, 3),
        link_pos=torch.tensor([[[0, 0, 0], [1.0, 0, 0]], [[0, 0, 0], [1.0, 0, 0]]]),
        link_lin_vel_w=torch.zeros(2, 2, 3),
    )
    sensor = RootAngularMomentumSensorCfg(name="L").build()
    sensor.bind(env)
    out = sensor.data
    assert out.shape == (2, 3)
    assert torch.all(out == 0.0)


def test_root_angular_momentum_single_link_orbital_formula() -> None:
    """One link at r=(1, 0, 0) with v=(0, 1, 0), mass=2 ⇒ L = m·r×v = (0, 0, 2)."""
    env = _FakeAngMomEnv(
        num_envs=1,
        link_names=("base", "arm"),
        # mass of the root link is irrelevant (r=0 ⇒ cross=0); only ``arm`` contributes.
        masses=torch.tensor([1.0, 2.0]),
        root_pos=torch.zeros(1, 3),
        link_pos=torch.tensor([[[0, 0, 0], [1.0, 0, 0]]]),
        link_lin_vel_w=torch.tensor([[[0, 0, 0], [0.0, 1.0, 0.0]]]),
    )
    sensor = RootAngularMomentumSensorCfg(name="L").build()
    sensor.bind(env)
    out = sensor.data
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 2.0]]), atol=1e-6)


def test_root_angular_momentum_zero_when_get_mass_unavailable() -> None:
    """If the robot lacks ``get_links_inertial_mass``, the sensor falls back to zero L."""
    env = _FakeAngMomEnv(
        num_envs=1,
        link_names=("base",),
        masses=None,  # FakeMassRobot returns None
        root_pos=torch.zeros(1, 3),
        link_pos=torch.zeros(1, 1, 3),
        link_lin_vel_w=torch.tensor([[[1.0, 0.0, 0.0]]]),
    )
    sensor = RootAngularMomentumSensorCfg(name="L").build()
    sensor.bind(env)
    assert torch.all(sensor.data == 0.0)


# ---------------------------------------------------------------------- ForceTorqueSensor (M3.5)


class _FakeForceRobot:
    def __init__(self, force: torch.Tensor) -> None:  # (num_envs, total_dofs)
        self._force = force

    def get_dofs_force(self) -> torch.Tensor:
        return self._force


class _FakeArticulation:
    def __init__(self, actuated_dof_ids: torch.Tensor) -> None:
        self.actuated_dof_ids = actuated_dof_ids


class _FakeFTEnv:
    def __init__(
        self, joint_names: tuple[str, ...], actuated_dof_ids: list[int], force: torch.Tensor
    ) -> None:
        self.num_envs = force.shape[0]
        self.device = "cpu"
        self.joint_names = list(joint_names)
        self.articulation = _FakeArticulation(torch.tensor(actuated_dof_ids, dtype=torch.long))
        self.robot = _FakeForceRobot(force)
        self.sensors: dict[str, object] = {}


# 3 actuated joints (j0/j1/j2) sit at global DoFs 2/3/4 (DoFs 0–1 are a floating base).
_FT_FORCE = torch.tensor([[10.0, 11, 12, 13, 14], [20, 21, 22, 23, 24]])


def test_force_torque_sensor_all_joints_slices_actuated_dofs() -> None:
    env = _FakeFTEnv(("j0", "j1", "j2"), [2, 3, 4], _FT_FORCE)
    sensor = ForceTorqueSensorCfg(name="ft").build()
    sensor.bind(env)
    assert sensor.joint_names == ["j0", "j1", "j2"]
    assert torch.allclose(sensor.data.force, torch.tensor([[12.0, 13, 14], [22, 23, 24]]))


def test_force_torque_sensor_explicit_joint_subset() -> None:
    env = _FakeFTEnv(("j0", "j1", "j2"), [2, 3, 4], _FT_FORCE)
    sensor = ForceTorqueSensorCfg(name="ft", joint_names=("j1",)).build()
    sensor.bind(env)
    # j1 → global DoF 3.
    assert torch.allclose(sensor.data.force, torch.tensor([[13.0], [23.0]]))


def test_force_torque_sensor_regex_and_obs_term() -> None:
    env = _FakeFTEnv(("j0", "j1", "j2"), [2, 3, 4], _FT_FORCE)
    sensor = ForceTorqueSensorCfg(name="ft", joint_names_expr=r"j[02]").build()
    sensor.bind(env)
    env.sensors["ft"] = sensor
    # j0, j2 → global DoFs 2, 4. The obs term returns the same (B, N) tensor.
    expected = torch.tensor([[12.0, 14], [22, 24]])
    assert torch.allclose(sensor.data.force, expected)
    assert torch.allclose(joint_force_torque(env, "ft"), expected)


def test_force_torque_sensor_zeros_without_genesis_getter() -> None:
    env = _FakeFTEnv(("j0", "j1", "j2"), [2, 3, 4], _FT_FORCE)
    env.robot = object()  # no get_dofs_force
    sensor = ForceTorqueSensorCfg(name="ft").build()
    sensor.bind(env)
    assert torch.allclose(sensor.data.force, torch.zeros(2, 3))
