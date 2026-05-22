"""Init-order gate test for the BaseTermManager refactor (ROADMAP §9 R2.5 / ADR-0002).

ADR-0002 §Risks identifies the load-bearing risk for R2.5 as
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
hold today?") rather than aspirational. They ship in the same PR as
the R2.5 refactor (commit 1 = this test, commit 2 = the refactor)
so any future contributor catches an init-order regression at test
time, not at the next CUDA-sync surprise.
"""

from __future__ import annotations

from typing import Any

import torch

from genelab.managers import (
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
