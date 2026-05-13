"""Sensor abstraction + obs-noise pipeline. Uses a fake env so Genesis is not required."""

from dataclasses import dataclass
from typing import Any

import torch

from genelab.managers import (
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
)
from genelab.mdp.noise import Gnoise, Unoise
from genelab.sensor import BodyVelocitySensorCfg, Sensor, SensorCfg


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
    assert ((first.abs() <= 0.1 + 1e-6).all())
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
