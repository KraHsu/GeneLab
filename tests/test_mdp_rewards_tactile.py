"""Unit tests for ``genelab.mdp.rewards.tactile`` (contact_intensity_l2 / contact_count)."""

from dataclasses import dataclass
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.mdp.rewards.tactile import (  # noqa: E402
    contact_count,
    contact_intensity_l2,
    slip_penalty,
)


@dataclass
class _FakeTactileData:
    raw: torch.Tensor


class _FakeTactileSensor:
    def __init__(self, raw: torch.Tensor) -> None:
        self._data = _FakeTactileData(raw=raw)

    @property
    def data(self) -> _FakeTactileData:
        return self._data


class _FakeEnv:
    def __init__(self, sensors: dict[str, Any]) -> None:
        self.sensors = sensors


def test_contact_intensity_l2_sum_of_squares_per_env() -> None:
    raw = torch.tensor([[1.0, -2.0, 0.5], [0.0, 0.0, 0.1]])  # (2, 3)
    env = _FakeEnv({"t": _FakeTactileSensor(raw)})
    reward = contact_intensity_l2(env, "t")  # type: ignore[arg-type]
    expected = torch.tensor([1.0 + 4.0 + 0.25, 0.0 + 0.0 + 0.01])
    assert torch.allclose(reward, expected)
    assert reward.shape == (2,)


def test_contact_intensity_l2_flattens_extra_axes() -> None:
    raw = torch.zeros(2, 3, 4)
    raw[0, 0, 0] = 2.0
    raw[1, 2, 3] = -1.0
    env = _FakeEnv({"t": _FakeTactileSensor(raw)})
    reward = contact_intensity_l2(env, "t")  # type: ignore[arg-type]
    assert reward.shape == (2,)
    assert torch.allclose(reward, torch.tensor([4.0, 1.0]))


def test_contact_count_thresholded() -> None:
    raw = torch.tensor([[0.0, 0.5, -1.0, 0.01], [2.0, -2.0, 0.0, 0.1]])
    env = _FakeEnv({"t": _FakeTactileSensor(raw)})
    counts = contact_count(env, "t", threshold=0.05)  # type: ignore[arg-type]
    # env 0: 0.5, 1.0 above threshold → 2; env 1: 2.0, 2.0, 0.1 above → 3.
    assert torch.allclose(counts, torch.tensor([2.0, 3.0]))


def test_contact_count_default_threshold_zero() -> None:
    raw = torch.tensor([[0.0, 0.5, 0.0], [1.0, 1.0, 1.0]])
    env = _FakeEnv({"t": _FakeTactileSensor(raw)})
    counts = contact_count(env, "t")  # type: ignore[arg-type]
    assert torch.allclose(counts, torch.tensor([1.0, 3.0]))


def test_slip_penalty_squares_lateral_components() -> None:
    """Last axis is xyz; the xy slice is the tangential plane."""
    raw = torch.tensor(
        [
            [[1.0, 2.0, 9.0], [0.0, 0.0, 5.0]],  # env 0: lateral (1, 2) + (0, 0) → 5
            [[0.0, 0.0, 1.0], [3.0, 4.0, 0.0]],  # env 1: lateral (0, 0) + (3, 4) → 25
        ]
    )
    env = _FakeEnv({"t": _FakeTactileSensor(raw)})
    reward = slip_penalty(env, "t")  # type: ignore[arg-type]
    assert torch.allclose(reward, torch.tensor([5.0, 25.0]))


def test_slip_penalty_works_with_history_axis() -> None:
    """``(B, H, P, 3)`` shape: lateral slice still ignores the z channel; history axis is reduced."""
    raw = torch.zeros(2, 3, 1, 3)
    raw[0, 0, 0] = torch.tensor([3.0, 0.0, 0.0])  # contributes 9
    raw[0, 2, 0] = torch.tensor([0.0, 4.0, 100.0])  # contributes 16 (z ignored)
    raw[1, 1, 0] = torch.tensor([2.0, 2.0, 5.0])  # contributes 8
    env = _FakeEnv({"t": _FakeTactileSensor(raw)})
    reward = slip_penalty(env, "t")  # type: ignore[arg-type]
    assert torch.allclose(reward, torch.tensor([25.0, 8.0]))
