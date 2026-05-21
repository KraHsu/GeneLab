"""Shared algorithm taxonomy for the RL backends.

Both :mod:`genelab.rl.sb3_config` and :mod:`genelab.rl.skrl_config` previously
declared their own copies of these two frozensets verbatim. They are the same
taxonomy — on-policy algorithms learn from a rollout buffer, off-policy
algorithms learn from a replay buffer — so per ADR-0003 / R2.1 they live here
and the configs re-export them.

Adding a new algorithm symbol means editing exactly one file. Each backend
config still keeps its own ``<Backend>Algorithm`` Literal so the type-checker
can keep narrowing per-backend.
"""

from __future__ import annotations

__all__ = [
    "OFF_POLICY_ALGORITHMS",
    "ON_POLICY_ALGORITHMS",
]

#: Algorithms that learn from a rollout buffer (SB3 calls the size ``n_steps``;
#: skrl calls it ``rollouts``).
ON_POLICY_ALGORITHMS: frozenset[str] = frozenset({"PPO", "A2C"})

#: Algorithms that learn from a replay buffer. SB3's HER replay buffer wraps
#: these three; skrl's ``replay_buffer_size`` configures the equivalent.
OFF_POLICY_ALGORITHMS: frozenset[str] = frozenset({"SAC", "TD3", "DDPG"})
