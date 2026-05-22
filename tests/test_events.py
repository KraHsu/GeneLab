"""Unit tests for event-term functions added in P1."""

from dataclasses import dataclass

import torch

from genelab.mdp.events import reset_joints_by_offset


class _FakeArticulation:
    """Captures the most recent ``write_joint_state`` call so tests can inspect it."""

    def __init__(
        self,
        joint_pos_limits: torch.Tensor | None = None,
        default_joint_pos: torch.Tensor | None = None,
    ) -> None:
        self.last_pos: torch.Tensor | None = None
        self.last_vel: torch.Tensor | None = None
        self.last_env_ids: torch.Tensor | None = None
        # Empty tensor signals "no limits configured" — matches the real
        # Articulation's stub state before bind, and disables the clamp.
        self.joint_pos_limits = (
            joint_pos_limits if joint_pos_limits is not None else torch.empty(0, 2)
        )
        # M3.6 S3b: event terms read default_joint_pos via the articulation (asset-routed),
        # mirroring the real env where env.default_joint_pos delegates to articulation.
        self.default_joint_pos = (
            default_joint_pos if default_joint_pos is not None else torch.zeros(0)
        )

    def write_joint_state(
        self, pos: torch.Tensor, vel: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        self.last_pos = pos.clone()
        self.last_vel = vel.clone()
        self.last_env_ids = env_ids.clone()


@dataclass
class _FakeEnv:
    default_joint_pos: torch.Tensor
    articulation: _FakeArticulation
    device: str = "cpu"


def _make_env(n_joints: int = 4) -> _FakeEnv:
    default = torch.arange(n_joints, dtype=torch.float)
    return _FakeEnv(
        default_joint_pos=default,
        articulation=_FakeArticulation(default_joint_pos=default),
    )


def test_reset_joints_by_offset_zero_range_matches_default_pose() -> None:
    env = _make_env(n_joints=4)
    env_ids = torch.tensor([0, 2])
    reset_joints_by_offset(env, env_ids, position_range=(0.0, 0.0), velocity_range=(0.0, 0.0))
    assert env.articulation.last_pos is not None
    assert env.articulation.last_vel is not None
    expected_pos = env.default_joint_pos.unsqueeze(0).expand(2, -1)
    assert torch.equal(env.articulation.last_pos, expected_pos)
    assert torch.equal(env.articulation.last_vel, torch.zeros(2, 4))
    assert torch.equal(env.articulation.last_env_ids, env_ids)


def test_reset_joints_by_offset_samples_within_position_range() -> None:
    torch.manual_seed(0)
    env = _make_env(n_joints=8)
    env_ids = torch.arange(64)  # large N stabilises range checks
    lo, hi = -0.3, 0.5
    reset_joints_by_offset(env, env_ids, position_range=(lo, hi), velocity_range=(0.0, 0.0))
    assert env.articulation.last_pos is not None
    offsets = env.articulation.last_pos - env.default_joint_pos.unsqueeze(0)
    assert torch.all(offsets >= lo)
    assert torch.all(offsets <= hi)
    # Distribution should span the interval — not all zero, both signs present.
    assert (offsets > 0).any() and (offsets < 0).any()


def test_reset_joints_by_offset_velocity_range_applies_independently() -> None:
    torch.manual_seed(0)
    env = _make_env(n_joints=6)
    env_ids = torch.arange(32)
    reset_joints_by_offset(env, env_ids, position_range=(0.0, 0.0), velocity_range=(-1.0, 1.0))
    assert env.articulation.last_vel is not None
    vel = env.articulation.last_vel
    assert torch.all(vel >= -1.0) and torch.all(vel <= 1.0)
    assert (vel.abs() > 1e-4).any()
    # Positions should be exactly default since position_range is zero.
    expected_pos = env.default_joint_pos.unsqueeze(0).expand(32, -1)
    assert env.articulation.last_pos is not None
    assert torch.equal(env.articulation.last_pos, expected_pos)


def test_reset_joints_by_offset_empty_env_ids_is_noop() -> None:
    env = _make_env(n_joints=4)
    reset_joints_by_offset(env, env_ids=None)
    assert env.articulation.last_pos is None
    reset_joints_by_offset(env, env_ids=torch.tensor([], dtype=torch.long))
    assert env.articulation.last_pos is None


def test_reset_joints_by_offset_clamps_to_joint_limits() -> None:
    """mjlab parity: large position offsets must be clipped to the joint limits."""
    torch.manual_seed(0)
    # 2 joints, tight limits: joint 0 ∈ [-0.05, 0.05], joint 1 ∈ [-0.1, 0.1].
    limits = torch.tensor([[-0.05, 0.05], [-0.1, 0.1]])
    env = _FakeEnv(
        default_joint_pos=torch.zeros(2),
        articulation=_FakeArticulation(joint_pos_limits=limits, default_joint_pos=torch.zeros(2)),
    )
    env_ids = torch.arange(32)
    # Sample range much wider than limits → every value should saturate within.
    reset_joints_by_offset(env, env_ids, position_range=(-1.0, 1.0), velocity_range=(0.0, 0.0))
    pos = env.articulation.last_pos
    assert pos is not None
    assert torch.all(pos[:, 0] >= -0.05) and torch.all(pos[:, 0] <= 0.05)
    assert torch.all(pos[:, 1] >= -0.1) and torch.all(pos[:, 1] <= 0.1)


def test_reset_joints_by_offset_no_clamp_when_limits_empty() -> None:
    """Stub limits (numel=0) means the clamp is skipped — used by tests/fakes."""
    torch.manual_seed(1)
    env = _make_env(n_joints=4)  # default fake has empty limits tensor
    env_ids = torch.arange(64)
    reset_joints_by_offset(env, env_ids, position_range=(-2.0, 2.0), velocity_range=(0.0, 0.0))
    pos = env.articulation.last_pos
    assert pos is not None
    # No clamp → can see values outside ±π (the fake has no limits enforcement).
    assert (pos.abs() > 0.5).any()
