"""Unit tests for the deterministic evaluator.

These tests cover ``run_evaluation`` directly with a fake :class:`InferenceSetup`
(no Genesis runtime needed) so the success-rate accounting, episode-budget
termination, and ``is_success`` plumbing can be verified deterministically.

End-to-end CLI tests that actually train + eval through the backends require a
Genesis runtime; those live as integration tests outside this file (run via the
verification block in the M1 plan).
"""

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.rl.backends.base import InferenceSetup  # noqa: E402
from genelab.rl.evaluator import EvalConfig, run_evaluation  # noqa: E402


class _FakeAdapter:
    """Vectorized fake env: per-env episode-length tracker + scripted terminations.

    On ``step()`` returns ``(obs, reward=1.0, dones, info)`` where ``dones[i]`` is
    True every ``period_per_env[i]`` steps. Optionally publishes ``is_success`` in
    ``info`` when ``per_step_success`` is supplied (per-step bool tensor of shape
    ``(num_envs,)``).
    """

    def __init__(
        self,
        num_envs: int,
        period_per_env: list[int],
        per_step_success: list[list[bool]] | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.period_per_env = list(period_per_env)
        self.per_step_success = per_step_success
        self._step_idx = 0
        self._step_in_episode = torch.zeros(num_envs, dtype=torch.long)
        self._obs = torch.zeros(num_envs, 3)

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        self._step_in_episode.zero_()
        self._step_idx = 0
        return self._obs, {}

    def step(
        self, _actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        self._step_in_episode += 1
        rewards = torch.ones(self.num_envs)
        dones = torch.tensor(
            [
                self._step_in_episode[i].item() >= self.period_per_env[i]
                for i in range(self.num_envs)
            ],
            dtype=torch.bool,
        )
        # Reset finished episodes' counters for next step
        for i in range(self.num_envs):
            if dones[i]:
                self._step_in_episode[i] = 0
        info: dict[str, Any] = {}
        if self.per_step_success is not None and self._step_idx < len(self.per_step_success):
            info["is_success"] = torch.tensor(
                self.per_step_success[self._step_idx], dtype=torch.bool
            )
        self._step_idx += 1
        return self._obs, rewards, dones, info


def _fake_policy(_obs: Any) -> torch.Tensor:
    return torch.zeros(_obs.shape[0], 2)


def _setup(adapter: _FakeAdapter) -> InferenceSetup:
    return InferenceSetup(
        wrapper=adapter,
        adapter=adapter,
        policy=_fake_policy,
        actor_module=None,
        actor_input_dim=None,
    )


def test_run_evaluation_collects_target_episodes() -> None:
    # 4 envs, each episode lasts 5 steps -> episode return = 5.0
    adapter = _FakeAdapter(num_envs=4, period_per_env=[5, 5, 5, 5])
    result = run_evaluation(_setup(adapter), EvalConfig(num_envs=4, episodes=8), num_envs=4)
    assert result.num_episodes == 8
    assert result.metrics.return_mean == pytest.approx(5.0)
    assert result.metrics.length_mean == pytest.approx(5.0)
    assert result.metrics.success_rate is None


def test_run_evaluation_success_rate_when_signal_present() -> None:
    # 2 envs, episodes of length 3, custom success tensor at each step
    per_step_success = [
        [False, False],  # step 0
        [False, False],  # step 1
        [True, False],  # step 2 — env 0 ends with success, env 1 with failure
    ]
    adapter = _FakeAdapter(num_envs=2, period_per_env=[3, 3], per_step_success=per_step_success)
    result = run_evaluation(_setup(adapter), EvalConfig(num_envs=2, episodes=2), num_envs=2)
    assert result.num_episodes == 2
    assert result.metrics.success_rate == pytest.approx(0.5)


def test_run_evaluation_success_rate_none_when_signal_absent() -> None:
    adapter = _FakeAdapter(num_envs=2, period_per_env=[3, 3], per_step_success=None)
    result = run_evaluation(_setup(adapter), EvalConfig(num_envs=2, episodes=2), num_envs=2)
    assert result.metrics.success_rate is None


def test_run_evaluation_respects_max_steps_cap() -> None:
    # Episode period 100, but max_steps=3 -> no complete episode -> num_episodes=0
    adapter = _FakeAdapter(num_envs=2, period_per_env=[100, 100])
    result = run_evaluation(
        _setup(adapter),
        EvalConfig(num_envs=2, episodes=10, max_steps=3),
        num_envs=2,
    )
    assert result.num_episodes == 0
    assert result.metrics.return_mean == 0.0


def test_eval_cli_payload_schema_matches_roadmap() -> None:
    """Smoke-check the JSON shape ``eval_task`` would emit, built directly from EvalResult.

    This is the documented eval-result schema; downstream tools (best-model
    selection in :mod:`genelab.rl.eval_callback`, reference-runs docs) read these
    keys verbatim, so the shape is load-bearing.
    """
    from genelab.rl.evaluator import EvalMetrics, EvalResult

    result = EvalResult(
        metrics=EvalMetrics(
            return_mean=12.3,
            return_std=1.0,
            length_mean=50.0,
            success_rate=None,
        ),
        num_episodes=4,
        wall_clock_seconds=0.5,
    )
    from dataclasses import asdict

    payload = {
        "task": "GeneLab-Inverted-Pendulum-v0",
        "checkpoint": "logs/x/model_1.pt",
        "num_episodes": result.num_episodes,
        "metrics": asdict(result.metrics),
        "wall_clock_seconds": result.wall_clock_seconds,
        "seed": 0,
        "deterministic": True,
    }
    assert payload["metrics"]["success_rate"] is None
    assert set(payload["metrics"].keys()) == {
        "return_mean",
        "return_std",
        "length_mean",
        "success_rate",
    }
    assert payload["num_episodes"] == 4
