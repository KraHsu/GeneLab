"""Smoke tests for the slim manager port. Uses a fake env so Genesis is not required."""

from typing import Any

import torch

from genelab.managers import (
    CommandTerm,
    CommandTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
    RewardManager,
    RewardTermCfg,
    TerminationManager,
    TerminationTermCfg,
)
from genelab.managers.command_manager import CommandManager
from genelab.managers.event_manager import EventManager


class _FakeEnv:
    """Just enough surface area for the managers to access env state."""

    def __init__(self, num_envs: int = 4, device: str = "cpu") -> None:
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length_s = 10.0


def _const_obs(value: float, dim: int):
    def fn(env: Any) -> torch.Tensor:
        return torch.full((env.num_envs, dim), value, device=env.device)

    return fn


def _per_env_reward(values: list[float]):
    def fn(env: Any) -> torch.Tensor:
        return torch.tensor(values, device=env.device)

    return fn


def _all_true(env: Any) -> torch.Tensor:
    return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)


def _all_false(env: Any) -> torch.Tensor:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def test_observation_manager_concats_terms_in_declared_order() -> None:
    env = _FakeEnv(num_envs=2)
    cfg = {
        "policy": ObservationGroupCfg(
            terms={
                "a": ObservationTermCfg(func=_const_obs(1.0, 3)),
                "b": ObservationTermCfg(func=_const_obs(2.0, 2), scale=0.5),
                "c": ObservationTermCfg(func=_const_obs(99.0, 1), clip=(0.0, 5.0)),
            }
        )
    }
    mgr = ObservationManager(cfg, env)
    obs = mgr.compute()
    assert "policy" in obs
    assert obs["policy"].shape == (2, 6)
    # a=[1,1,1], b=[2*0.5, 2*0.5], c=[clip(99,0,5)=5]
    assert torch.allclose(obs["policy"][0], torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 5.0]))
    assert mgr.group_obs_dim("policy") == 6


def test_reward_manager_weights_and_sums_and_scales_by_dt() -> None:
    env = _FakeEnv(num_envs=3)
    cfg = {
        "alpha": RewardTermCfg(func=_per_env_reward([1.0, 2.0, 3.0]), weight=2.0),
        "beta": RewardTermCfg(func=_per_env_reward([0.5, 0.5, 0.5]), weight=-1.0),
    }
    mgr = RewardManager(cfg, env)
    out = mgr.compute(dt=0.1)
    expected = torch.tensor(
        [(1.0 * 2.0 + 0.5 * -1.0) * 0.1] * 0
        + [
            (1.0 * 2.0 - 0.5) * 0.1,
            (2.0 * 2.0 - 0.5) * 0.1,
            (3.0 * 2.0 - 0.5) * 0.1,
        ]
    )
    assert torch.allclose(out, expected, atol=1e-6)


def test_termination_manager_separates_dones_and_time_outs() -> None:
    env = _FakeEnv(num_envs=2)
    cfg = {
        "time_out": TerminationTermCfg(func=_all_true, time_out=True),
        "fall": TerminationTermCfg(func=_all_false, time_out=False),
    }
    mgr = TerminationManager(cfg, env)
    dones = mgr.compute()
    assert torch.equal(dones, torch.tensor([True, True]))
    assert torch.equal(mgr.time_outs, torch.tensor([True, True]))
    assert torch.equal(mgr.terminated, torch.tensor([False, False]))


def test_event_manager_dispatches_by_mode() -> None:
    env = _FakeEnv(num_envs=2)
    called = {"startup": 0, "reset": 0, "interval": 0}

    def startup_fn(_env, _env_ids):
        called["startup"] += 1

    def reset_fn(_env, _env_ids):
        called["reset"] += 1

    def interval_fn(_env, _env_ids):
        called["interval"] += 1

    cfg = {
        "boot": EventTermCfg(mode="startup", func=startup_fn),
        "reset_pose": EventTermCfg(mode="reset", func=reset_fn),
        "push": EventTermCfg(mode="interval", interval_range_s=(0.01, 0.01), func=interval_fn),
    }
    mgr = EventManager(cfg, env)
    mgr.apply("startup")
    assert called["startup"] == 1
    mgr.apply("reset", torch.arange(2))
    assert called["reset"] == 1
    # First interval tick may not fire yet because time_left was randomized in [0.01, 0.01]
    mgr.apply("interval", dt=0.02)
    assert called["interval"] >= 1


class _DummyCommand(CommandTerm):
    def __init__(self, cfg: CommandTermCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self._cmd = torch.zeros(env.num_envs, 1, device=env.device)
        self.resampled = 0

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.resampled += 1
        self._cmd[env_ids] = float(self.resampled)


def test_command_manager_resamples_after_time_window() -> None:
    env = _FakeEnv(num_envs=2)
    cfg = {"twist": CommandTermCfg(class_type=_DummyCommand, resampling_time_range=(1.0, 1.0))}
    mgr = CommandManager(cfg, env)
    # Force initial resample via reset
    mgr.reset(torch.arange(2))
    assert mgr._terms["twist"].resampled == 1
    # Step once with dt < time_left -> no resample
    mgr.compute(dt=0.5)
    assert mgr._terms["twist"].resampled == 1
    # Step past the window -> resample
    mgr.compute(dt=0.6)
    assert mgr._terms["twist"].resampled == 2
