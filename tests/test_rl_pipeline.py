"""Smoke tests for the RL config dataclasses + the RSL-RL VecEnv wrapper attribute surface."""

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.rl import (  # noqa: E402  (after importorskip)
    RslRlBaseRunnerCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from genelab.rl.vecenvs.rsl_rl import RslRlVecEnvWrapper  # noqa: E402


def test_rsl_rl_runner_cfg_defaults_match_mjlab_shape() -> None:
    cfg = RslRlOnPolicyRunnerCfg()
    assert cfg.class_name == "OnPolicyRunner"
    assert cfg.num_steps_per_env == 24
    assert cfg.actor.hidden_dims == (128, 128, 128)
    assert cfg.actor.activation == "elu"
    assert cfg.actor.distribution_cfg is not None
    assert cfg.actor.distribution_cfg["class_name"] == "GaussianDistribution"
    assert cfg.critic.hidden_dims == (128, 128, 128)
    assert cfg.algorithm.gamma == 0.99
    assert cfg.algorithm.lam == 0.95
    assert cfg.algorithm.clip_param == 0.2
    # Tensorboard default (we don't ship wandb)
    assert cfg.logger == "tensorboard"


def test_runner_cfg_is_a_subclass_of_base() -> None:
    cfg = RslRlOnPolicyRunnerCfg()
    assert isinstance(cfg, RslRlBaseRunnerCfg)
    assert isinstance(cfg.algorithm, RslRlPpoAlgorithmCfg)


class _FakeActionManager:
    total_action_dim = 4


class _FakeObsManager:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs

    def compute(self) -> dict[str, Any]:
        return {
            "policy": torch.zeros(self.num_envs, 7),
            "critic": torch.zeros(self.num_envs, 7),
        }


class _FakeEnv:
    def __init__(self) -> None:
        self.num_envs = 4
        self.device = "cpu"
        self.max_episode_length = 100
        self.action_manager = _FakeActionManager()
        self.observation_manager = _FakeObsManager(self.num_envs)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.cfg = object()

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


def test_rsl_rl_wrapper_exposes_runner_attrs() -> None:
    env = _FakeEnv()
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    assert wrapped.num_envs == 4
    assert wrapped.num_actions == 4
    assert wrapped.num_obs == 7
    assert wrapped.num_privileged_obs == 7
    obs = wrapped.get_observations()
    # When tensordict is installed it should be a TensorDict; otherwise a dict.
    assert hasattr(obs, "__getitem__")
    # Reset returns (obs, extras)
    out, extras = wrapped.reset()
    assert hasattr(out, "__getitem__")
    assert isinstance(extras, dict)
    # Step returns 4-tuple matching RSL-RL VecEnv contract.
    obs2, rew, dones, info = wrapped.step(torch.zeros(env.num_envs, 4))
    assert rew.shape == (env.num_envs,)
    assert dones.shape == (env.num_envs,)
    assert "time_outs" in info


def test_play_task_caps_scripted_agent_at_simulation_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scripted (zero/random) playback defaults its step cap to ``simulation.steps``
    (what ``--steps`` sets) so a headless ``play --agent zero --steps N`` stops after N
    steps; trained playback stays unbounded unless ``max_steps`` is passed."""
    from types import SimpleNamespace

    from genelab.rl import runner

    captured: dict[str, object] = {}

    class _FakeBackend:
        name = "fake"

        def play(self, ctx: Any) -> None:
            captured["max_steps"] = ctx.max_steps

    monkeypatch.setattr(runner, "ensure_project_cache", lambda: None)
    monkeypatch.setattr(runner, "build_env", lambda cfg: object())
    monkeypatch.setattr(runner, "build_bridges", lambda cfg: [])
    monkeypatch.setattr(runner, "default_backend", lambda: _FakeBackend())
    monkeypatch.setattr(runner, "TASKS", SimpleNamespace(get=lambda _id: None))

    cfg = SimpleNamespace(simulation=SimpleNamespace(steps=7, num_envs=1))

    runner.play_task("Unregistered-Task-v0", env_cfg=cfg, agent="zero")
    assert captured["max_steps"] == 7

    from pathlib import Path

    runner.play_task(
        "Unregistered-Task-v0", env_cfg=cfg, agent="trained", checkpoint=Path("x.pt")
    )
    assert captured["max_steps"] is None
