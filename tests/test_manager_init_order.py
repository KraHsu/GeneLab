"""Init-order gate test for the BaseTermManager refactor.

The load-bearing risk for this refactor is
*init-order preservation*: after we pull ``RewardManager.__init__`` and
``TerminationManager.__init__`` registration loops up to
``BaseTermManager``, the subclass-specific buffer allocations
(currently inline at the bottom of each ``__init__``) must still
happen **after** term registration runs. If the order flipped —
buffers allocated before terms are registered — ``_episode_sums``
and ``_term_dones`` would be empty dicts and the manager's
``reset()`` / ``compute()`` methods would silently no-op for the
per-term tallies.

These assertions are deliberately observational ("does the invariant
hold today?") rather than aspirational. They ship alongside the refactor
so any future contributor catches an init-order regression at test
time, not at the next CUDA-sync surprise.
"""

from __future__ import annotations

from typing import Any

import torch

from genelab.managers import (
    CurriculumManager,
    CurriculumTermCfg,
    MetricsManager,
    MetricsTermCfg,
    RewardManager,
    RewardTermCfg,
    TerminationManager,
    TerminationTermCfg,
)


class _FakeEnv:
    """Minimal env surface — mirrors the pattern from tests/test_managers.py."""

    def __init__(self, num_envs: int = 4, device: str = "cpu") -> None:
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length_s = 10.0


def _per_env_reward(values: list[float]):
    def fn(env: Any) -> torch.Tensor:
        return torch.tensor(values, device=env.device)

    return fn


def _bool_signal(value: bool):
    def fn(env: Any) -> torch.Tensor:
        return torch.full((env.num_envs,), value, dtype=torch.bool, device=env.device)

    return fn


def test_reward_manager_post_init_allocates_episode_sums() -> None:
    """RewardManager.__init__ leaves ``_episode_sums`` populated with per-env zero tensors."""
    env = _FakeEnv(num_envs=4)
    cfg = {
        "alpha": RewardTermCfg(func=_per_env_reward([1.0, 2.0, 3.0, 4.0]), weight=1.0),
        "beta": RewardTermCfg(func=_per_env_reward([0.5] * 4), weight=2.0),
    }
    mgr = RewardManager(cfg, env)

    # Term registration ran: _term_names matches cfg order.
    assert mgr._term_names == ["alpha", "beta"]
    assert [type(c).__name__ for c in mgr._term_cfgs] == ["RewardTermCfg", "RewardTermCfg"]

    # Buffer allocation ran AFTER registration: _episode_sums keys mirror _term_names.
    assert set(mgr._episode_sums.keys()) == {"alpha", "beta"}
    for name, buf in mgr._episode_sums.items():
        assert buf.shape == (4,), f"_episode_sums[{name!r}].shape == {buf.shape}"
        assert buf.device.type == "cpu"
        assert buf.dtype == torch.float
        assert torch.all(buf == 0.0)

    # _reward_buf is per-env zero, ready for compute() to populate.
    assert mgr._reward_buf.shape == (4,)
    assert mgr._reward_buf.dtype == torch.float
    assert torch.all(mgr._reward_buf == 0.0)


def test_termination_manager_post_init_allocates_term_dones() -> None:
    """TerminationManager.__init__ leaves ``_term_dones`` populated with per-env zero bool tensors."""
    env = _FakeEnv(num_envs=3)
    cfg = {
        "time_out": TerminationTermCfg(func=_bool_signal(False), time_out=True),
        "fall": TerminationTermCfg(func=_bool_signal(False), time_out=False),
    }
    mgr = TerminationManager(cfg, env)

    # Term registration ran: _term_names matches cfg order.
    assert mgr._term_names == ["time_out", "fall"]
    assert [type(c).__name__ for c in mgr._term_cfgs] == [
        "TerminationTermCfg",
        "TerminationTermCfg",
    ]

    # Buffer allocation ran AFTER registration: _term_dones keys mirror _term_names.
    assert set(mgr._term_dones.keys()) == {"time_out", "fall"}
    for name, buf in mgr._term_dones.items():
        assert buf.shape == (3,), f"_term_dones[{name!r}].shape == {buf.shape}"
        assert buf.dtype == torch.bool
        assert not buf.any()

    # _truncated_buf and _terminated_buf are allocated, both per-env bool zeros.
    assert mgr._truncated_buf.shape == (3,)
    assert mgr._truncated_buf.dtype == torch.bool
    assert not mgr._truncated_buf.any()
    assert mgr._terminated_buf.shape == (3,)
    assert mgr._terminated_buf.dtype == torch.bool
    assert not mgr._terminated_buf.any()


def test_empty_cfg_yields_empty_per_term_dicts() -> None:
    """Empty cfg → empty per-term dicts (still dicts, not None), and the per-env buffers still allocated."""
    env = _FakeEnv(num_envs=2)

    rmgr = RewardManager(cfg={}, env=env)
    assert rmgr._term_names == []
    assert rmgr._episode_sums == {}
    assert rmgr._reward_buf.shape == (2,)

    tmgr = TerminationManager(cfg={}, env=env)
    assert tmgr._term_names == []
    assert tmgr._term_dones == {}
    assert tmgr._truncated_buf.shape == (2,)
    assert tmgr._terminated_buf.shape == (2,)


def _scalar_metric(values: list[float]):
    def fn(env: Any) -> torch.Tensor:
        return torch.tensor(values, device=env.device)

    return fn


def test_metrics_manager_post_init_allocates_episode_sums() -> None:
    """MetricsManager.__init__ leaves ``_episode_sums`` + ``_step_count`` populated post-registration."""
    env = _FakeEnv(num_envs=4)
    cfg = {
        "speed": MetricsTermCfg(func=_scalar_metric([1.0, 2.0, 3.0, 4.0])),
        "height": MetricsTermCfg(func=_scalar_metric([0.5] * 4)),
    }
    mgr = MetricsManager(cfg, env)

    # Term registration ran: _term_names matches cfg order.
    assert mgr._term_names == ["speed", "height"]
    assert [type(c).__name__ for c in mgr._term_cfgs] == ["MetricsTermCfg", "MetricsTermCfg"]

    # Buffer allocation ran AFTER registration: _episode_sums keys mirror _term_names.
    assert set(mgr._episode_sums.keys()) == {"speed", "height"}
    for name, buf in mgr._episode_sums.items():
        assert buf.shape == (4,), f"_episode_sums[{name!r}].shape == {buf.shape}"
        assert buf.device.type == "cpu"
        assert buf.dtype == torch.float
        assert torch.all(buf == 0.0)

    # _step_count is per-env long zeros, ready for compute() to increment.
    assert mgr._step_count.shape == (4,)
    assert mgr._step_count.dtype == torch.long
    assert torch.all(mgr._step_count == 0)


def test_curriculum_manager_registers_terms() -> None:
    """CurriculumManager.__init__ registers terms in cfg order; no per-term buffers."""
    env = _FakeEnv(num_envs=3)
    cfg = {
        "level_a": CurriculumTermCfg(func=lambda env, env_ids: None),
        "level_b": CurriculumTermCfg(func=lambda env, env_ids: None),
    }
    mgr = CurriculumManager(cfg, env)

    assert mgr._term_names == ["level_a", "level_b"]
    assert [type(c).__name__ for c in mgr._term_cfgs] == [
        "CurriculumTermCfg",
        "CurriculumTermCfg",
    ]
    # active_terms returns a copy of the registered names.
    assert mgr.active_terms == ["level_a", "level_b"]
    assert mgr.active_terms is not mgr._term_names


def test_metrics_curriculum_empty_cfg_yields_empty_per_term_state() -> None:
    """Empty cfg → empty per-term state (still containers, not None); metrics buffers still allocated."""
    env = _FakeEnv(num_envs=2)

    mmgr = MetricsManager(cfg={}, env=env)
    assert mmgr._term_names == []
    assert mmgr._episode_sums == {}
    assert mmgr._step_count.shape == (2,)

    cmgr = CurriculumManager(cfg={}, env=env)
    assert cmgr._term_names == []
    assert cmgr._term_cfgs == []
    assert cmgr.active_terms == []
