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


def test_rsl_rl_runner_cfg_defaults_match_reference_shape() -> None:
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


# -- recurrent (RNN) policy config -------------------------------------------------


def test_rnn_type_auto_selects_rnn_model() -> None:
    """Setting ``rnn_type`` alone flips ``class_name`` to ``RNNModel`` (one knob)."""
    from genelab.rl import RslRlModelCfg

    assert RslRlModelCfg().class_name == "MLPModel"  # default stays MLP
    for rnn_type in ("lstm", "gru"):
        cfg = RslRlModelCfg(rnn_type=rnn_type)
        assert cfg.class_name == "RNNModel"


def test_rnn_type_validation_and_explicit_class_name() -> None:
    from genelab.rl import RslRlModelCfg

    with pytest.raises(ValueError, match="rnn_type"):
        RslRlModelCfg(rnn_type="transformer")
    # An explicit non-default class_name is left untouched.
    assert RslRlModelCfg(rnn_type="gru", class_name="CNNModel").class_name == "CNNModel"
    # RNNModel without an rnn_type is rejected early (would otherwise crash in rsl_rl).
    with pytest.raises(ValueError, match="requires rnn_type"):
        RslRlModelCfg(class_name="RNNModel")


def test_runner_cfg_to_dict_prunes_per_model_class() -> None:
    """MLP drops every recurrent/cnn key; RNN keeps the rnn keys but still drops cnn_cfg."""
    from genelab.rl import RslRlModelCfg
    from genelab.rl.backends.rsl_rl import _runner_cfg_to_dict

    mlp = _runner_cfg_to_dict(RslRlOnPolicyRunnerCfg())["actor"]
    assert "rnn_type" not in mlp and "cnn_cfg" not in mlp

    rnn_cfg = RslRlModelCfg(rnn_type="lstm", rnn_hidden_dim=64, rnn_num_layers=2)
    rnn = _runner_cfg_to_dict(RslRlOnPolicyRunnerCfg(actor=rnn_cfg, critic=rnn_cfg))["actor"]
    assert rnn["class_name"] == "RNNModel"
    assert rnn["rnn_type"] == "lstm"
    assert rnn["rnn_hidden_dim"] == 64 and rnn["rnn_num_layers"] == 2
    assert "cnn_cfg" not in rnn  # RNNModel.__init__ rejects it


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


# -- recurrent hidden-state reset hook (play / eval) ------------------------------


class _ResetSpyEnv:
    """Minimal env/wrapper pair for the rollout loops; step emits a fixed ``dones``."""

    viewer_closed = False

    def __init__(self, dones: "torch.Tensor") -> None:
        self._dones = dones

    def reset(self):
        return torch.zeros(1, 3), {}

    def step(self, _actions):
        return torch.zeros(1, 3), torch.zeros(len(self._dones)), self._dones, {"time_outs": None}


def test_run_play_loop_resets_hidden_state_with_dones() -> None:
    """The play loop forwards each step's ``dones`` to ``reset_hidden`` (recurrent policies)."""
    from genelab.rl._helpers import run_play_loop

    dones = torch.tensor([True, False])
    env = _ResetSpyEnv(dones)
    seen: list[Any] = []
    run_play_loop(
        env,
        env,
        policy=lambda _obs: torch.zeros(1, 3),
        bridges=[],
        max_steps=3,
        prof_step=None,
        reset_hidden=lambda d: seen.append(d),
    )
    assert len(seen) == 3
    assert all(torch.equal(d, dones) for d in seen)


def test_run_play_loop_without_reset_hidden_is_a_noop() -> None:
    """A stateless policy (``reset_hidden=None``) runs the loop unchanged."""
    from genelab.rl._helpers import run_play_loop

    env = _ResetSpyEnv(torch.tensor([False]))
    run_play_loop(
        env, env, policy=lambda _obs: torch.zeros(1, 3), bridges=[], max_steps=2, prof_step=None
    )  # no reset_hidden kwarg -> must not raise


def test_run_play_loop_reset_hidden_allows_inplace_on_inference_tensor() -> None:
    """Regression: a recurrent reset zeroes the hidden state in place, but ``policy()``
    created that tensor as an *inference tensor* inside the loop's ``inference_mode``.
    The loop must run ``reset_hidden`` inside inference mode too — otherwise PyTorch raises
    'Inplace update to inference tensor outside InferenceMode is not allowed'. This mirrors
    rsl_rl's ``RNNModel.reset`` and is what surfaced in live ``genelab play``."""
    from genelab.rl._helpers import run_play_loop

    state: dict[str, Any] = {}

    def policy(_obs: Any) -> Any:
        # Runs under run_play_loop's inference_mode -> an inference tensor, like the
        # RNN hidden state created during the forward pass.
        state["h"] = torch.zeros(2)
        return torch.zeros(1, 3)

    def reset_hidden(dones: Any) -> None:
        state["h"][dones.view(-1).bool()] = 0.0  # in-place update on the inference tensor

    env = _ResetSpyEnv(torch.tensor([True, False]))
    run_play_loop(
        env, env, policy=policy, bridges=[], max_steps=2, prof_step=None, reset_hidden=reset_hidden
    )  # must not raise


def test_run_evaluation_resets_hidden_on_episode_boundary() -> None:
    """``run_evaluation`` calls ``reset_hidden(dones)`` every step so finished episodes
    start the next one from a zero hidden state."""
    from genelab.rl.backends.base import InferenceSetup
    from genelab.rl.evaluator import EvalConfig, run_evaluation

    # Each step ends the episode (done=True), so 2 steps collect 2 episodes.
    class _Adapter:
        def reset(self):
            return torch.zeros(1, 3), {}

        def step(self, _actions):
            return torch.zeros(1, 3), torch.zeros(1), torch.ones(1, dtype=torch.bool), {}

    seen: list[Any] = []
    setup = InferenceSetup(
        wrapper=None,
        adapter=_Adapter(),
        policy=lambda _obs: torch.zeros(1, 3),
        is_recurrent=True,
        reset_hidden=lambda d: seen.append(d),
    )
    run_evaluation(setup, EvalConfig(episodes=2, max_steps=10), num_envs=1)
    assert len(seen) >= 2  # at least one reset per completed episode


def test_play_task_step_cap_gates_on_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Playback length is gated on the viewer, not the agent kind: headless
    (``vis=False``) playback caps at ``simulation.steps`` for every kind (so a headless
    ``play --agent zero/trained`` can't hang); with a viewer (``vis=True``) it stays
    unbounded so it runs until the window is closed. An explicit ``max_steps`` wins."""
    from pathlib import Path
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

    def _cfg(vis: bool) -> SimpleNamespace:
        return SimpleNamespace(simulation=SimpleNamespace(steps=7, num_envs=1, vis=vis))

    # Headless: every kind is capped at simulation.steps.
    runner.play_task("Unregistered-Task-v0", env_cfg=_cfg(vis=False), agent="zero")
    assert captured["max_steps"] == 7
    runner.play_task(
        "Unregistered-Task-v0", env_cfg=_cfg(vis=False), agent="trained", checkpoint=Path("x.pt")
    )
    assert captured["max_steps"] == 7

    # Viewer on: unbounded — playback runs until the window closes.
    runner.play_task("Unregistered-Task-v0", env_cfg=_cfg(vis=True), agent="zero")
    assert captured["max_steps"] is None
    runner.play_task(
        "Unregistered-Task-v0", env_cfg=_cfg(vis=True), agent="trained", checkpoint=Path("x.pt")
    )
    assert captured["max_steps"] is None

    # Explicit max_steps always wins, viewer or not.
    runner.play_task("Unregistered-Task-v0", env_cfg=_cfg(vis=True), agent="zero", max_steps=3)
    assert captured["max_steps"] == 3
