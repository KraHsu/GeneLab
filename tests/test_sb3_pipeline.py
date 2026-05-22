"""Smoke tests for the SB3 backend: config defaults, the env wrapper surface
(flat + goal-conditioned HER modes), and backend selection.

The ``genelab.rl.vecenvs.sb3`` import is deliberately kept *inside* the test
functions rather than at module scope: importing it pulls in ``stable_baselines3``
(and transitively ``cv2``, which hijacks ``QT_QPA_PLATFORM_PLUGIN_PATH``). pytest
imports every test module during collection, so a module-level import would poison
the Qt plugin path for the PyQt-plotter integration tests that run earlier. The
wrapper degrades to a plain object without SB3, so these tests need no
``importorskip`` — they exercise GeneLab's own adapter logic either way.
"""

from typing import Any

import pytest

torch = pytest.importorskip("torch")

import gymnasium  # noqa: E402
import numpy as np  # noqa: E402

from genelab.rl import RslRlOnPolicyRunnerCfg, Sb3AgentCfg, Sb3HerCfg  # noqa: E402
from genelab.rl.backends import select_backend  # noqa: E402


class _FakeActionManager:
    total_action_dim = 4


class _FakeObsManager:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs

    def compute(self) -> dict[str, Any]:
        return {
            "policy": torch.zeros(self.num_envs, 7),
            "critic": torch.zeros(self.num_envs, 11),
            "achieved_goal": torch.zeros(self.num_envs, 3),
            "desired_goal": torch.zeros(self.num_envs, 3),
        }


class _FakeEnv:
    def __init__(self) -> None:
        self.num_envs = 4
        self.device = "cpu"
        self.max_episode_length = 100
        self.action_manager = _FakeActionManager()
        self.observation_manager = _FakeObsManager(self.num_envs)

    def reset(self):
        return self.observation_manager.compute(), {}

    def step(self, _actions):
        obs = self.observation_manager.compute()
        rew = torch.zeros(self.num_envs)
        term = torch.zeros(self.num_envs, dtype=torch.bool)
        trunc = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rew, term, trunc, {}

    def close(self) -> None:
        pass


def test_sb3_agent_cfg_defaults() -> None:
    cfg = Sb3AgentCfg()
    assert cfg.algorithm == "PPO"
    assert cfg.is_on_policy is True
    assert Sb3AgentCfg(algorithm="SAC").is_on_policy is False
    assert cfg.obs_group == "policy"
    assert cfg.policy.net_arch == (256, 256)
    assert cfg.her.enabled is False


def test_select_backend_routes_by_cfg_type() -> None:
    assert select_backend(Sb3AgentCfg()).name == "sb3"
    assert select_backend(RslRlOnPolicyRunnerCfg()).name == "rsl_rl"
    with pytest.raises(ValueError):
        select_backend(object())


def test_sb3_wrapper_flat_contract() -> None:
    from genelab.rl.vecenvs.sb3 import GenelabSb3VecEnv

    env = _FakeEnv()
    wrapped = GenelabSb3VecEnv(env, obs_group="policy")

    assert isinstance(wrapped.observation_space, gymnasium.spaces.Box)
    assert wrapped.observation_space.shape == (7,)
    assert isinstance(wrapped.action_space, gymnasium.spaces.Box)
    assert wrapped.action_space.shape == (4,)
    assert wrapped.num_envs == 4

    obs = wrapped.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4, 7)

    obs2, rewards, dones, infos = wrapped.step(torch.zeros(4, 4))
    assert obs2.shape == (4, 7)
    assert rewards.shape == (4,)
    assert dones.shape == (4,) and dones.dtype == bool
    assert isinstance(infos, list) and len(infos) == 4


def test_sb3_wrapper_rejects_missing_obs_group() -> None:
    from genelab.rl.vecenvs.sb3 import GenelabSb3VecEnv

    with pytest.raises(KeyError):
        GenelabSb3VecEnv(_FakeEnv(), obs_group="does-not-exist")


def test_sb3_wrapper_her_contract() -> None:
    from genelab.rl.vecenvs.sb3 import GenelabSb3VecEnv

    captured: dict[str, Any] = {}

    def _compute_reward(achieved_goal, desired_goal, info):  # noqa: ANN001
        captured["call"] = (achieved_goal, desired_goal, info)
        return np.zeros(len(achieved_goal), dtype=np.float32)

    her = Sb3HerCfg(enabled=True, compute_reward=_compute_reward)
    wrapped = GenelabSb3VecEnv(_FakeEnv(), her_cfg=her)

    assert isinstance(wrapped.observation_space, gymnasium.spaces.Dict)
    assert set(wrapped.observation_space.spaces) == {
        "observation",
        "achieved_goal",
        "desired_goal",
    }

    obs = wrapped.reset()
    assert isinstance(obs, dict)
    assert obs["observation"].shape == (4, 7)
    assert obs["achieved_goal"].shape == (4, 3)
    assert obs["desired_goal"].shape == (4, 3)

    obs2, _rewards, _dones, _infos = wrapped.step(torch.zeros(4, 4))
    assert isinstance(obs2, dict)

    ag = np.zeros((4, 3), dtype=np.float32)
    dg = np.ones((4, 3), dtype=np.float32)
    result = wrapped.env_method("compute_reward", ag, dg, [{}] * 4, indices=[0])
    assert len(result) == 1
    assert result[0].shape == (4,)
    assert "call" in captured
