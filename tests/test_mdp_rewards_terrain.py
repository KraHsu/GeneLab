"""Unit tests for soft-terrain reward terms (``genelab.mdp.rewards.terrain``, paper §8.3).

Genesis-free: the rewards read the privileged deformable-terrain state, stubbed here with
``SimpleNamespace`` so the cost functions are exercised through their public signature.
"""

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.mdp.rewards.terrain import footprint_revisit, terrain_sinkage_l2  # noqa: E402


def _env(depth: Any = None, residual: Any = None) -> Any:
    if depth is None and residual is None:
        return SimpleNamespace(deformable_terrain=None)
    state = SimpleNamespace(depth=depth, residual=residual)
    driver = SimpleNamespace(terrain=SimpleNamespace(state=state))
    return SimpleNamespace(deformable_terrain=driver)


def test_terrain_sinkage_l2_penalises_squared_foot_depth() -> None:
    env = _env(depth=torch.tensor([[0.02, 0.03]]))  # 2 feet, one env
    cost = terrain_sinkage_l2(env)
    assert float(cost) == pytest.approx(0.02**2 + 0.03**2)


def test_footprint_revisit_penalises_standing_on_residual() -> None:
    env = _env(residual=torch.tensor([[0.01, 0.04]]))
    cost = footprint_revisit(env)
    assert float(cost) == pytest.approx(0.05)  # Σ residual


def test_terrain_sinkage_l2_ignores_feet_above_the_surface() -> None:
    env = _env(depth=torch.tensor([[-0.05, 0.02]]))  # first foot above surface
    cost = terrain_sinkage_l2(env)
    assert float(cost) == pytest.approx(0.02**2)  # negative depth clamped out


def test_soft_terrain_rewards_are_per_env() -> None:
    env = _env(depth=torch.tensor([[0.1, 0.0], [0.0, 0.2]]))
    cost = terrain_sinkage_l2(env)
    assert cost.shape == (2,)
    assert torch.allclose(cost, torch.tensor([0.01, 0.04]))


def test_soft_terrain_rewards_require_configured_terrain() -> None:
    env = _env()  # deformable_terrain = None
    with pytest.raises(RuntimeError):
        terrain_sinkage_l2(env)
    with pytest.raises(RuntimeError):
        footprint_revisit(env)
