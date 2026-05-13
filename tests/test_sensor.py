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
from genelab.sensor import Sensor, SensorCfg


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
