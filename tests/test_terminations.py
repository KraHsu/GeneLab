"""Unit tests for termination term functions (M2.3).

A minimal fake env provides ``joint_pos_limits`` (the per-actuated-joint
``(lower, upper)`` table the real env exposes via ``Articulation.joint_pos_limits``)
and ``robot_state.joint_pos`` — enough for the limit termination. No Genesis runtime.
"""

from dataclasses import dataclass

import torch

from genelab.mdp.terminations import joint_pos_out_of_limit


@dataclass
class _FakeRobotState:
    joint_pos: torch.Tensor


@dataclass
class _FakeEnv:
    robot_state: _FakeRobotState
    joint_pos_limits: torch.Tensor  # (J, 2)


def _env(joint_pos: list[list[float]], limits: list[tuple[float, float]]) -> _FakeEnv:
    return _FakeEnv(
        robot_state=_FakeRobotState(joint_pos=torch.tensor(joint_pos, dtype=torch.float)),
        joint_pos_limits=torch.tensor(limits, dtype=torch.float),
    )


def test_joint_pos_out_of_limit_flags_only_out_of_bounds_envs() -> None:
    # Two joints, limits [-1, 1] each. Env 0 in-bounds, env 1 over upper, env 2 under lower.
    env = _env(
        joint_pos=[[0.5, -0.5], [0.5, 1.5], [-2.0, 0.0]],
        limits=[(-1.0, 1.0), (-1.0, 1.0)],
    )
    out = joint_pos_out_of_limit(env)
    assert out.dtype == torch.bool
    assert out.shape == (3,)
    assert out.tolist() == [False, True, True]


def test_joint_pos_out_of_limit_at_the_limit_is_not_out() -> None:
    # Exactly at the boundary counts as in-bounds (strict < / > comparison).
    env = _env(joint_pos=[[-1.0, 1.0]], limits=[(-1.0, 1.0), (-1.0, 1.0)])
    assert joint_pos_out_of_limit(env).tolist() == [False]


def test_joint_pos_out_of_limit_ignores_infinite_limit_joints() -> None:
    # A continuous joint (±inf limits) never trips, regardless of position.
    env = _env(
        joint_pos=[[1e6, 0.0], [0.0, 5.0]],
        limits=[(float("-inf"), float("inf")), (-1.0, 1.0)],
    )
    assert joint_pos_out_of_limit(env).tolist() == [False, True]
