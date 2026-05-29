"""Unit tests for ``genelab.mdp.rewards.energy``."""

from dataclasses import dataclass
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.mdp.rewards.energy import (  # noqa: E402
    energy_budget,
    kinetic_energy_l2,
    potential_energy,
)


@dataclass
class _FakeGsHandle:
    """Minimal stand-in for a Genesis robot handle with the energy methods rewards read."""

    kinetic: torch.Tensor
    potential: torch.Tensor

    def get_kinetic_energy(self) -> torch.Tensor:
        return self.kinetic

    def get_potential_energy(self) -> torch.Tensor:
        return self.potential

    def get_total_energy(self) -> torch.Tensor:
        return self.kinetic + self.potential


class _FakeArticulation:
    def __init__(self, handle: _FakeGsHandle) -> None:
        self.gs_handle = handle


class _FakeEnv:
    def __init__(self, articulations: dict[str, _FakeArticulation]) -> None:
        self.articulations = articulations
        # ``asset_handle`` falls back to ``env.robot`` when ``asset_cfg`` is None and
        # the entity isn't in ``articulations``; expose the primary by default.
        self.robot: Any = articulations["robot"].gs_handle


def _env(*, kinetic: torch.Tensor, potential: torch.Tensor) -> _FakeEnv:
    return _FakeEnv({"robot": _FakeArticulation(_FakeGsHandle(kinetic, potential))})


def test_kinetic_energy_l2_squares_per_env() -> None:
    env = _env(kinetic=torch.tensor([0.0, 1.5, -2.0]), potential=torch.zeros(3))
    reward = kinetic_energy_l2(env)  # type: ignore[arg-type]
    assert torch.allclose(reward, torch.tensor([0.0, 2.25, 4.0]))


def test_potential_energy_returns_signed_value() -> None:
    env = _env(kinetic=torch.zeros(2), potential=torch.tensor([-9.81, 4.2]))
    reward = potential_energy(env)  # type: ignore[arg-type]
    assert torch.allclose(reward, torch.tensor([-9.81, 4.2]))


def test_energy_budget_squares_deviation_from_target() -> None:
    env = _env(kinetic=torch.tensor([1.0, 2.0]), potential=torch.tensor([2.0, -1.0]))
    # total = (3.0, 1.0); target=2.0 → deltas (1.0, -1.0) → squared (1.0, 1.0)
    reward = energy_budget(env, target_total=2.0)  # type: ignore[arg-type]
    assert torch.allclose(reward, torch.tensor([1.0, 1.0]))


def test_energy_budget_default_target_zero() -> None:
    env = _env(kinetic=torch.tensor([1.0, 3.0]), potential=torch.tensor([0.0, -1.0]))
    # total = (1.0, 2.0); target=0.0 → squared (1.0, 4.0)
    reward = energy_budget(env)  # type: ignore[arg-type]
    assert torch.allclose(reward, torch.tensor([1.0, 4.0]))
