"""Numerical-equivalence gate for the motion-tracking parameterization.

The three near-identical body-error rewards
(`motion_relative_body_position_error_exp`,
`motion_global_body_linear_velocity_error_exp`,
`motion_global_body_angular_velocity_error_exp`) were collapsed onto the shared
`motion_body_error_exp(..., quantity=...)` factory. These tests pin the *original*
pre-refactor implementations and assert the factory (and its public wrappers)
reproduce them bit-for-bit, with and without the `body_names` filter.

No Genesis runtime needed — a duck-typed fake `MotionCommand` supplies just the
attributes the rewards read (`_motion_command` does a no-op `cast`).
"""

import torch

from genelab.mdp.motion_tracking import (
    motion_body_error_exp,
    motion_global_body_angular_velocity_error_exp,
    motion_global_body_linear_velocity_error_exp,
    motion_relative_body_position_error_exp,
)


class _FakeCfg:
    def __init__(self, body_names: tuple[str, ...]) -> None:
        self.body_names = body_names


class _FakeMotionCommand:
    def __init__(self, n_bodies: int, batch: int, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.cfg = _FakeCfg(tuple(f"body_{i}" for i in range(n_bodies)))
        shape = (batch, n_bodies, 3)
        self.body_pos_relative_w = torch.rand(shape, generator=gen)
        self.robot_body_pos_w = torch.rand(shape, generator=gen)
        self.body_lin_vel_w = torch.rand(shape, generator=gen)
        self.robot_body_lin_vel_w = torch.rand(shape, generator=gen)
        self.body_ang_vel_w = torch.rand(shape, generator=gen)
        self.robot_body_ang_vel_w = torch.rand(shape, generator=gen)


class _FakeCommandManager:
    def __init__(self, cmd: _FakeMotionCommand) -> None:
        self._terms = {"motion": cmd}


class _FakeEnv:
    def __init__(self, cmd: _FakeMotionCommand) -> None:
        self.command_manager = _FakeCommandManager(cmd)


def _index_filter(cmd: _FakeMotionCommand, body_names: tuple[str, ...] | None) -> list[int]:
    return [
        i for i, name in enumerate(cmd.cfg.body_names) if body_names is None or name in body_names
    ]


# --- Reference implementations: verbatim copies of the pre-refactor function bodies. -----


def _ref_position(env, command_name, std, body_names=None):
    cmd = env.command_manager._terms[command_name]
    indexes = _index_filter(cmd, body_names)
    error = torch.sum(
        (cmd.body_pos_relative_w[:, indexes] - cmd.robot_body_pos_w[:, indexes]) ** 2,
        dim=-1,
    )
    return torch.exp(-error.mean(dim=-1) / (std * std))


def _ref_linear_velocity(env, command_name, std, body_names=None):
    cmd = env.command_manager._terms[command_name]
    indexes = _index_filter(cmd, body_names)
    error = torch.sum(
        (cmd.body_lin_vel_w[:, indexes] - cmd.robot_body_lin_vel_w[:, indexes]) ** 2,
        dim=-1,
    )
    return torch.exp(-error.mean(dim=-1) / (std * std))


def _ref_angular_velocity(env, command_name, std, body_names=None):
    cmd = env.command_manager._terms[command_name]
    indexes = _index_filter(cmd, body_names)
    error = torch.sum(
        (cmd.body_ang_vel_w[:, indexes] - cmd.robot_body_ang_vel_w[:, indexes]) ** 2,
        dim=-1,
    )
    return torch.exp(-error.mean(dim=-1) / (std * std))


_CASES = [
    ("pos", motion_relative_body_position_error_exp, _ref_position),
    ("lin_vel", motion_global_body_linear_velocity_error_exp, _ref_linear_velocity),
    ("ang_vel", motion_global_body_angular_velocity_error_exp, _ref_angular_velocity),
]


def test_wrappers_match_reference_implementations() -> None:
    env = _FakeEnv(_FakeMotionCommand(n_bodies=5, batch=8, seed=0))
    for quantity, wrapper, reference in _CASES:
        expected = reference(env, "motion", 0.5)
        # Public wrapper and the directly-called factory must both equal the original.
        assert torch.equal(wrapper(env, "motion", 0.5), expected), quantity
        assert torch.equal(
            motion_body_error_exp(env, "motion", 0.5, quantity=quantity), expected
        ), quantity


def test_wrappers_match_reference_with_body_filter() -> None:
    env = _FakeEnv(_FakeMotionCommand(n_bodies=5, batch=8, seed=1))
    subset = ("body_0", "body_3")
    for quantity, wrapper, reference in _CASES:
        expected = reference(env, "motion", 0.7, subset)
        assert torch.equal(wrapper(env, "motion", 0.7, subset), expected), quantity
        assert torch.equal(
            motion_body_error_exp(env, "motion", 0.7, subset, quantity=quantity), expected
        ), quantity


def test_factory_quantity_selects_distinct_signals() -> None:
    # Sanity: the three quantities read different attribute pairs, so on random
    # state they should not collapse to the same tensor (guards a copy-paste swap).
    env = _FakeEnv(_FakeMotionCommand(n_bodies=4, batch=6, seed=2))
    pos = motion_body_error_exp(env, "motion", 0.5, quantity="pos")
    lin = motion_body_error_exp(env, "motion", 0.5, quantity="lin_vel")
    ang = motion_body_error_exp(env, "motion", 0.5, quantity="ang_vel")
    assert not torch.equal(pos, lin)
    assert not torch.equal(pos, ang)
    assert not torch.equal(lin, ang)
