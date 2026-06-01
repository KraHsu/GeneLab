"""Registration smoke test for the inverted-pendulum extension."""

from pathlib import Path
from typing import Any

import pytest

from genelab.registry import ENVS, ROBOTS, TASKS, load_extension_module

SKRL_TASK_ID = "GeneLab-Inverted-Pendulum-Skrl-v0"


def test_inverted_pendulum_extension_registers() -> None:
    load_extension_module("genelab_inverted_pendulum.tasks")

    assert "GeneLab-Inverted-Pendulum-v0" in TASKS.names()
    assert "GeneLab-Double-Inverted-Pendulum-v0" in TASKS.names()
    assert SKRL_TASK_ID in TASKS.names()

    assert "inverted-pendulum" in ROBOTS.names()
    assert "double-inverted-pendulum" in ROBOTS.names()

    assert "inverted-pendulum-env" in ENVS.names()
    assert "double-inverted-pendulum-env" in ENVS.names()

    # "Recall the target" memory showcase: RNN + MLP baseline on a shared env.
    assert "GeneLab-Inverted-Pendulum-Memory-Rnn-v0" in TASKS.names()
    assert "GeneLab-Inverted-Pendulum-Memory-Mlp-v0" in TASKS.names()
    assert "inverted-pendulum-memory-env" in ENVS.names()


def test_memory_task_rnn_is_recurrent_mlp_is_not() -> None:
    """The memory showcase task carries an LSTM ``RslRlModelCfg`` (auto-selected ``RNNModel``);
    its baseline stays a feed-forward ``MLPModel``. Both share the memory env and a rollout
    length that spans the episode (so truncated BPTT can learn the recall)."""
    load_extension_module("genelab_inverted_pendulum.tasks")

    rnn = TASKS.get("GeneLab-Inverted-Pendulum-Memory-Rnn-v0")
    assert rnn.cfg.agent.actor.rnn_type == "lstm"
    assert rnn.cfg.agent.actor.class_name == "RNNModel"
    assert rnn.cfg.agent.num_steps_per_env == 100  # BPTT spans a ~100-step episode

    mlp = TASKS.get("GeneLab-Inverted-Pendulum-Memory-Mlp-v0")
    assert mlp.cfg.agent.actor.class_name == "MLPModel"
    assert rnn.cfg.env_name == mlp.cfg.env_name == "inverted-pendulum-memory-env"
    assert rnn.cfg.robot_name == "inverted-pendulum"


def test_memory_env_cfg_has_masked_target_and_tracking_reward() -> None:
    """The memory env replaces the cart-centring reward with hidden-target tracking, adds the
    (masked) target-cue observation, and re-samples the target each reset — the pieces that
    make the task require memory."""
    from genelab_inverted_pendulum.single import inverted_pendulum_memory_env_cfg

    cfg = inverted_pendulum_memory_env_cfg()
    assert "target_cue" in cfg.observations_cfg["policy"].terms
    assert "cart_target" in cfg.rewards_cfg
    assert "cart_position" not in cfg.rewards_cfg  # replaced by target tracking
    assert "reset_cart_target" in cfg.events_cfg
    assert cfg.episode_length_s == 1.0  # short episodes keep the dependency BPTT-learnable


def test_inverted_pendulum_skrl_task_uses_skrl_agent_cfg() -> None:
    """The skrl task reuses the single-pendulum env but carries a ``SkrlAgentCfg``,
    which is what routes it to the skrl backend (and what ``genelab info`` shows)."""
    from genelab.rl import SkrlAgentCfg

    load_extension_module("genelab_inverted_pendulum.tasks")
    task = TASKS.get(SKRL_TASK_ID)
    assert isinstance(task.cfg.agent, SkrlAgentCfg)
    assert task.cfg.agent.algorithm == "PPO"
    assert task.cfg.trainable is True
    # Reuses the single-pendulum env / robot (no new env registered).
    assert task.cfg.env_name == "inverted-pendulum-env"
    assert task.cfg.robot_name == "inverted-pendulum"


class _FakeObsManager:
    def compute(self) -> dict[str, Any]:
        import torch

        return {"policy": torch.zeros(4, 7), "critic": torch.zeros(4, 11)}


class _FakeActionManager:
    total_action_dim = 1  # single inverted pendulum drives one cart actuator


class _FakeEnv:
    """Minimal env satisfying the skrl wrapper + trainer contract (no Genesis)."""

    num_envs = 4
    device = "cpu"
    max_episode_length = 100

    def __init__(self) -> None:
        self.observation_manager = _FakeObsManager()
        self.action_manager = _FakeActionManager()

    def reset(self) -> Any:
        return self.observation_manager.compute(), {}

    def step(self, _actions: Any) -> Any:
        import torch

        obs = self.observation_manager.compute()
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, torch.zeros(self.num_envs), done, done, {}

    def close(self) -> None:
        pass


def test_inverted_pendulum_skrl_train_eval_roundtrip(tmp_path: Path) -> None:
    """Drive the real skrl backend with the task's cfg: a short train writes a
    checkpoint (final-save guarantees one even at ``timesteps=2``), and the eval
    inference setup loads it back into a runnable policy. Skipped without skrl."""
    pytest.importorskip("torch")
    pytest.importorskip("skrl")
    import torch

    from genelab.rl.backends.base import PlayContext, ProfileArgs, TrainContext
    from genelab.rl.backends.skrl import SkrlBackend
    from genelab_inverted_pendulum.single import inverted_pendulum_skrl_agent_cfg

    cfg = inverted_pendulum_skrl_agent_cfg()
    backend = SkrlBackend()
    backend.train(
        TrainContext(
            task_id=SKRL_TASK_ID,
            env=_FakeEnv(),
            env_cfg=None,
            agent_cfg=cfg,
            max_iterations=2,
            seed=0,
            log_dir=tmp_path,
        )
    )
    checkpoints = sorted((tmp_path / "checkpoints").glob("*.pt"))
    assert checkpoints, "skrl train wrote no checkpoint"

    setup = backend.make_inference_setup(
        PlayContext(
            task_id=SKRL_TASK_ID,
            env=_FakeEnv(),
            env_cfg=None,
            agent_cfg=cfg,
            checkpoint=checkpoints[0],
            kind="trained",
            deterministic=True,
            max_steps=None,
            bridges=[],
            profile=ProfileArgs(),
        )
    )
    actions = setup.policy(torch.zeros(4, 7))
    assert actions.shape[0] == 4
