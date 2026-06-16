"""Smoke tests for the skrl backend: config defaults, the env wrapper surface,
backend selection, and the per-algorithm model factory.

Skipped wholesale when skrl is not installed (the backend itself stays importable).
"""

from typing import Any

import pytest

torch = pytest.importorskip("torch")
skrl = pytest.importorskip("skrl")  # noqa: F841

import gymnasium  # noqa: E402
import torch.nn as nn  # noqa: E402

from genelab.rl import RslRlOnPolicyRunnerCfg, SkrlAgentCfg  # noqa: E402
from genelab.rl.backends import select_backend  # noqa: E402
from genelab.rl.backends.skrl import _build_agent  # noqa: E402  # pyright: ignore[reportPrivateUsage]
from genelab.rl.backends.skrl.config import SkrlModelCfg  # noqa: E402
from genelab.rl.backends.skrl.models import build_models  # noqa: E402
from genelab.rl.vecenvs.skrl import GenelabSkrlWrapper  # noqa: E402

ALGORITHMS = ["PPO", "A2C", "SAC", "TD3", "DDPG"]

_EXPECTED_MODEL_KEYS = {
    "PPO": {"policy", "value"},
    "A2C": {"policy", "value"},
    "DDPG": {"policy", "target_policy", "critic", "target_critic"},
    "TD3": {
        "policy",
        "target_policy",
        "critic_1",
        "critic_2",
        "target_critic_1",
        "target_critic_2",
    },
    "SAC": {"policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2"},
}


class _FakeActionManager:
    total_action_dim = 4


class _FakeObsManager:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs

    def compute(self) -> dict[str, Any]:
        return {
            "policy": torch.zeros(self.num_envs, 7),
            "critic": torch.zeros(self.num_envs, 11),
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


def test_skrl_agent_cfg_defaults() -> None:
    cfg = SkrlAgentCfg()
    assert cfg.algorithm == "PPO"
    assert cfg.is_on_policy is True
    assert SkrlAgentCfg(algorithm="SAC").is_on_policy is False
    assert cfg.obs_group == "policy"
    assert cfg.models.hidden_dims == (256, 256)


def test_select_backend_routes_by_cfg_type() -> None:
    assert select_backend(SkrlAgentCfg()).name == "skrl"
    assert select_backend(RslRlOnPolicyRunnerCfg()).name == "rsl_rl"
    with pytest.raises(ValueError):
        select_backend(object())


def test_skrl_wrapper_contract() -> None:
    env = _FakeEnv()
    wrapped = GenelabSkrlWrapper(env, obs_group="policy", state_group="critic")

    assert isinstance(wrapped.observation_space, gymnasium.spaces.Box)
    assert wrapped.observation_space.shape == (7,)
    assert isinstance(wrapped.action_space, gymnasium.spaces.Box)
    assert wrapped.action_space.shape == (4,)
    assert isinstance(wrapped.state_space, gymnasium.spaces.Box)
    assert wrapped.state_space.shape == (11,)
    assert wrapped.num_envs == 4
    assert wrapped.num_agents == 1

    obs, info = wrapped.reset()
    assert obs.shape == (4, 7)
    assert isinstance(info, dict)

    obs2, reward, terminated, truncated, info2 = wrapped.step(torch.zeros(4, 4))
    assert obs2.shape == (4, 7)
    # skrl expects reward / done flags shaped (num_envs, 1).
    assert reward.shape == (4, 1)
    assert terminated.shape == (4, 1)
    assert truncated.shape == (4, 1)


def test_skrl_wrapper_rejects_missing_obs_group() -> None:
    with pytest.raises(KeyError):
        GenelabSkrlWrapper(_FakeEnv(), obs_group="does-not-exist")


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_build_models_per_algorithm(algorithm: str) -> None:
    wrapped = GenelabSkrlWrapper(_FakeEnv())
    models = build_models(algorithm, wrapped, SkrlAgentCfg().models, device="cpu")
    assert set(models) == _EXPECTED_MODEL_KEYS[algorithm]


# --- off-policy stability fixes ------------------------------------------------
# Three changes make off-policy (SAC/TD3/DDPG) actually trainable instead of
# diverging: empirical obs normalization, a LayerNorm-stabilized Q-critic, and a
# non-clipping SAC policy (so the entropy term can't blow up the Q-target).


def test_obs_normalization_defaults_off() -> None:
    assert SkrlModelCfg().obs_normalization is False


def test_qcritic_is_layernorm_stabilized() -> None:
    # The Q-critic (SAC/TD3/DDPG only) gets LayerNorm to bound value extrapolation.
    wrapped = GenelabSkrlWrapper(_FakeEnv())
    models = build_models("SAC", wrapped, SkrlAgentCfg().models, device="cpu")
    q_has_ln = any(isinstance(m, nn.LayerNorm) for m in models["critic_1"].net.modules())
    assert q_has_ln
    # PPO uses the state-value critic, which must stay a plain MLP (no behaviour change).
    ppo = build_models("PPO", wrapped, SkrlAgentCfg().models, device="cpu")
    assert not any(isinstance(m, nn.LayerNorm) for m in ppo["value"].net.modules())


def test_sac_policy_does_not_clip_actions() -> None:
    # skrl clamps the action before taking its log-prob; clipping on would let a
    # drifting policy mean send -alpha*log_prob (and the Q-target) to infinity.
    wrapped = GenelabSkrlWrapper(_FakeEnv())
    policy = build_models("SAC", wrapped, SkrlAgentCfg().models, device="cpu")["policy"]
    assert policy._g_clip_actions is False  # skrl GaussianMixin flag


def test_obs_normalization_wires_state_preprocessor() -> None:
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    wrapped = GenelabSkrlWrapper(_FakeEnv(), state_group=None)
    off = _build_agent(
        SkrlAgentCfg(algorithm="SAC", state_group=None, learning_starts=0), wrapped, "cpu", None
    )
    assert not isinstance(off._state_preprocessor, RunningStandardScaler)
    on = _build_agent(
        SkrlAgentCfg(
            algorithm="SAC",
            state_group=None,
            learning_starts=0,
            models=SkrlModelCfg(obs_normalization=True),
        ),
        wrapped,
        "cpu",
        None,
    )
    assert isinstance(on._state_preprocessor, RunningStandardScaler)
