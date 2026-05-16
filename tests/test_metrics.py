"""Unit tests for ``MetricsManager`` + ``mean_action_acc`` (P5)."""

from dataclasses import dataclass

import torch

from genelab.managers import ActionTermCfg, MetricsManager, MetricsTermCfg
from genelab.managers.action_manager import ActionManager, ActionTerm
from genelab.mdp.metrics import mean_action_acc


# --------------------------------------------------------------------- fakes


@dataclass
class _FakeMetricsEnv:
    num_envs: int = 4
    device: str = "cpu"


def _const(value: float):
    """Return a metric function that yields ``(B,)`` filled with ``value`` for every env."""

    def f(env: _FakeMetricsEnv) -> torch.Tensor:
        return torch.full((env.num_envs,), value, device=env.device)

    return f


def _per_env(values: torch.Tensor):
    """Return a metric function yielding a fixed per-env vector."""

    def f(env: _FakeMetricsEnv) -> torch.Tensor:  # noqa: ARG001 - fixed values
        return values

    return f


# --------------------------------------------------------------------- MetricsManager


def test_metrics_manager_accumulates_per_env_sums_across_compute_calls() -> None:
    env = _FakeMetricsEnv(num_envs=2)
    cfg = {"const_one": MetricsTermCfg(func=_const(1.0))}
    mgr = MetricsManager(cfg, env)
    for _ in range(5):
        mgr.compute()
    # Five compute() calls × value 1.0 → sum 5.0 per env.
    assert torch.allclose(mgr._episode_sums["const_one"], torch.full((2,), 5.0))
    assert torch.equal(mgr._step_count, torch.full((2,), 5, dtype=torch.long))


def test_metrics_manager_reset_returns_per_term_episode_mean() -> None:
    env = _FakeMetricsEnv(num_envs=2)
    cfg = {"x": MetricsTermCfg(func=_const(2.5))}
    mgr = MetricsManager(cfg, env)
    for _ in range(4):
        mgr.compute()
    extras = mgr.reset()
    # Mean = sum/count = (2.5*4)/4 = 2.5 per env; averaged across envs → 2.5.
    assert extras["Episode_Metrics/x"] == 2.5


def test_metrics_manager_reset_clears_only_specified_envs() -> None:
    env = _FakeMetricsEnv(num_envs=4)
    cfg = {"x": MetricsTermCfg(func=_const(1.0))}
    mgr = MetricsManager(cfg, env)
    for _ in range(3):
        mgr.compute()
    mgr.reset(torch.tensor([0, 1]))
    # Reset envs 0,1 → their accumulators zeroed; envs 2,3 untouched.
    assert torch.equal(mgr._episode_sums["x"], torch.tensor([0.0, 0.0, 3.0, 3.0]))
    assert torch.equal(mgr._step_count, torch.tensor([0, 0, 3, 3], dtype=torch.long))


def test_metrics_manager_reset_with_zero_step_count_returns_finite_mean() -> None:
    """An env reset before any compute() call should report 0, not NaN."""
    env = _FakeMetricsEnv(num_envs=2)
    cfg = {"x": MetricsTermCfg(func=_const(7.0))}
    mgr = MetricsManager(cfg, env)
    # Skip compute(); reset immediately — clamp(min=1) guards the divide.
    extras = mgr.reset()
    assert extras["Episode_Metrics/x"] == 0.0


def test_metrics_manager_empty_cfg_is_a_no_op() -> None:
    env = _FakeMetricsEnv()
    mgr = MetricsManager({}, env)
    mgr.compute()  # no-op
    assert mgr.reset() == {}


def test_metrics_manager_distinguishes_per_env_values_at_reset() -> None:
    """Per-env metric values must survive the mean reduction across the env subset."""
    env = _FakeMetricsEnv(num_envs=4)
    per_env_value = torch.tensor([1.0, 2.0, 3.0, 4.0])
    cfg = {"x": MetricsTermCfg(func=_per_env(per_env_value))}
    mgr = MetricsManager(cfg, env)
    mgr.compute()  # single step → episode_sum equals per_env_value, count=1
    # Reset all → mean across all 4 envs = 2.5.
    extras = mgr.reset()
    assert abs(extras["Episode_Metrics/x"] - 2.5) < 1e-6


# --------------------------------------------------------------------- ActionManager history


class _NoopActionCfg(ActionTermCfg):
    pass


class _NoopActionTerm(ActionTerm):
    def __init__(self, cfg: ActionTermCfg, env) -> None:  # type: ignore[no-untyped-def]
        super().__init__(cfg, env)
        self._dim = 3

    @property
    def action_dim(self) -> int:
        return self._dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, self._dim)

    def process_actions(self, actions: torch.Tensor) -> None:  # noqa: ARG002
        pass

    def apply_actions(self) -> None:
        pass


def test_action_manager_three_step_history_shifts_correctly() -> None:
    """Three ``process_action`` calls leave the buffers ordered ``a / a_prev / a_prev_prev``."""
    env = _FakeMetricsEnv(num_envs=1)
    cfg = {"noop": _NoopActionCfg(class_type=_NoopActionTerm)}
    mgr = ActionManager(cfg, env)
    v0 = torch.tensor([[0.1, 0.2, 0.3]])
    v1 = torch.tensor([[1.0, 1.0, 1.0]])
    v2 = torch.tensor([[-0.5, 0.0, 0.5]])
    mgr.process_action(v0)
    mgr.process_action(v1)
    mgr.process_action(v2)
    assert torch.allclose(mgr.action, v2)
    assert torch.allclose(mgr.prev_action, v1)
    assert torch.allclose(mgr.prev_prev_action, v0)


def test_action_manager_reset_clears_all_three_buffers() -> None:
    env = _FakeMetricsEnv(num_envs=1)
    cfg = {"noop": _NoopActionCfg(class_type=_NoopActionTerm)}
    mgr = ActionManager(cfg, env)
    mgr.process_action(torch.tensor([[1.0, 2.0, 3.0]]))
    mgr.process_action(torch.tensor([[4.0, 5.0, 6.0]]))
    mgr.process_action(torch.tensor([[7.0, 8.0, 9.0]]))
    mgr.reset()
    assert torch.all(mgr.action == 0.0)
    assert torch.all(mgr.prev_action == 0.0)
    assert torch.all(mgr.prev_prev_action == 0.0)


# --------------------------------------------------------------------- mean_action_acc


class _FakeActionManager:
    """Surface the three history slots without going through ActionManager."""

    def __init__(self, a, prev, prev_prev) -> None:
        self.action = a
        self.prev_action = prev
        self.prev_prev_action = prev_prev


@dataclass
class _FakeActionAccEnv:
    action_manager: _FakeActionManager
    num_envs: int = 1
    device: str = "cpu"


def test_mean_action_acc_zero_when_history_is_constant() -> None:
    """A perfectly constant action stream has zero second derivative ⇒ metric=0."""
    a = torch.ones(2, 4)
    env = _FakeActionAccEnv(action_manager=_FakeActionManager(a, a, a), num_envs=2)
    out = mean_action_acc(env)
    assert torch.allclose(out, torch.zeros(2), atol=1e-6)


def test_mean_action_acc_known_finite_difference() -> None:
    """``a − 2·a_prev + a_prev_prev`` averaged over the action dim, per env."""
    # Env 0: action=[1,1,1], prev=[0,0,0], prev_prev=[0,0,0] → accel=[1,1,1] → mean=1.
    # Env 1: action=[0,0,2], prev=[0,1,1], prev_prev=[0,0,0] → accel=[0,-2,0] → mean=2/3.
    a = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 2.0]])
    p = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    pp = torch.zeros(2, 3)
    env = _FakeActionAccEnv(action_manager=_FakeActionManager(a, p, pp), num_envs=2)
    out = mean_action_acc(env)
    expected = torch.tensor([1.0, 2.0 / 3.0])
    assert torch.allclose(out, expected, atol=1e-6)
