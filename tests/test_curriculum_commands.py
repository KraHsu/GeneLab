"""Unit tests for ``mdp.commands_vel`` — staged velocity-range expansion curriculum."""

from dataclasses import dataclass

import torch

from genelab.mdp.commands.velocity_command import UniformVelocityCommandCfg
from genelab.mdp.curriculums import commands_vel


@dataclass
class _FakeCommandTerm:
    """Minimal term surface for the curriculum: only ``cfg.ranges`` is read."""

    cfg: UniformVelocityCommandCfg


class _FakeCommandManager:
    def __init__(self, term: _FakeCommandTerm) -> None:
        self._term = term

    def get_term(self, name: str) -> _FakeCommandTerm:  # noqa: ARG002 - single-term fake
        return self._term


@dataclass
class _FakeEnv:
    common_step_counter: int
    command_manager: _FakeCommandManager
    device: str = "cpu"


def _make_env(common_step: int = 0) -> tuple[_FakeEnv, UniformVelocityCommandCfg]:
    cfg = UniformVelocityCommandCfg(
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-0.5, 0.5),
        )
    )
    term = _FakeCommandTerm(cfg=cfg)
    env = _FakeEnv(
        common_step_counter=common_step,
        command_manager=_FakeCommandManager(term),
    )
    return env, cfg


_STAGES = [
    {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
    {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
    {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
]


def test_commands_vel_returns_initial_stage_at_step_zero() -> None:
    env, cfg = _make_env(common_step=0)
    out = commands_vel(env, env_ids=None, command_name="twist", velocity_stages=_STAGES)
    assert cfg.ranges.lin_vel_x == (-1.0, 1.0)
    assert cfg.ranges.ang_vel_z == (-0.5, 0.5)
    # The dict return is what the curriculum manager logs.
    assert isinstance(out, dict)
    assert torch.equal(out["lin_vel_x_min"], torch.tensor(-1.0))
    assert torch.equal(out["lin_vel_x_max"], torch.tensor(1.0))


def test_commands_vel_promotes_at_second_stage_threshold() -> None:
    env, cfg = _make_env(common_step=5000 * 24)
    commands_vel(env, env_ids=None, command_name="twist", velocity_stages=_STAGES)
    assert cfg.ranges.lin_vel_x == (-1.5, 2.0)
    assert cfg.ranges.ang_vel_z == (-0.7, 0.7)
    # lin_vel_y was untouched — inherits the initial range.
    assert cfg.ranges.lin_vel_y == (-1.0, 1.0)


def test_commands_vel_third_stage_inherits_previous_axes() -> None:
    """Third stage only sets ``lin_vel_x``; the others must keep stage-2 values."""
    env, cfg = _make_env(common_step=10000 * 24)
    commands_vel(env, env_ids=None, command_name="twist", velocity_stages=_STAGES)
    assert cfg.ranges.lin_vel_x == (-2.0, 3.0)
    assert cfg.ranges.ang_vel_z == (-0.7, 0.7)  # carried over from stage 2


def test_commands_vel_mutates_cfg_in_place_across_calls() -> None:
    """Repeated calls converge to the highest stage whose ``step`` <= counter."""
    env, cfg = _make_env(common_step=0)
    commands_vel(env, env_ids=None, command_name="twist", velocity_stages=_STAGES)
    assert cfg.ranges.lin_vel_x == (-1.0, 1.0)
    env.common_step_counter = 12000 * 24
    commands_vel(env, env_ids=None, command_name="twist", velocity_stages=_STAGES)
    assert cfg.ranges.lin_vel_x == (-2.0, 3.0)
