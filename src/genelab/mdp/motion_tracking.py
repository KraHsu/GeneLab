"""Motion-imitation reward terms (e.g. the Unitree G1 tracking task).

Factored out of ``mdp/rewards.py`` (ADR-0006 / ROADMAP §9 PR R5.1) so the
generic reward library stays a coherent "any task may use this" surface and the
motion-tracking family has one home. Consumers reach these via the
``genelab.mdp`` package namespace; ``mdp/rewards.py`` keeps a back-compat
re-export block so ``genelab.mdp.rewards.motion_*`` also still resolves.
"""

from typing import TYPE_CHECKING, Literal, cast

import torch

from genelab.mdp.commands.motion_command import MotionCommand
from genelab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _motion_command(env: "ManagerBasedRlEnv", command_name: str) -> MotionCommand:
    term = env.command_manager._terms[command_name]  # pyright: ignore[reportPrivateUsage]
    return cast(MotionCommand, term)


def _body_index_filter(cmd: MotionCommand, body_names: tuple[str, ...] | None) -> list[int]:
    return [
        i
        for i, name in enumerate(cmd.cfg.body_names)
        if (body_names is None) or (name in body_names)
    ]


def motion_global_anchor_position_error_exp(
    env: "ManagerBasedRlEnv", command_name: str, std: float
) -> torch.Tensor:
    """``exp(-||p_ref - p_robot||^2 / std^2)`` on the anchor body in world frame."""
    cmd = _motion_command(env, command_name)
    error = torch.sum((cmd.anchor_pos_w - cmd.robot_anchor_pos_w) ** 2, dim=-1)
    return torch.exp(-error / (std * std))


def motion_global_anchor_orientation_error_exp(
    env: "ManagerBasedRlEnv", command_name: str, std: float
) -> torch.Tensor:
    """Geodesic rotation error on the anchor body, mapped through a Gaussian kernel."""
    cmd = _motion_command(env, command_name)
    error = quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w) ** 2
    return torch.exp(-error / (std * std))


_BODY_ERROR_ATTRS: dict[str, tuple[str, str]] = {
    "pos": ("body_pos_relative_w", "robot_body_pos_w"),
    "lin_vel": ("body_lin_vel_w", "robot_body_lin_vel_w"),
    "ang_vel": ("body_ang_vel_w", "robot_body_ang_vel_w"),
}


def motion_body_error_exp(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
    *,
    quantity: Literal["pos", "lin_vel", "ang_vel"],
) -> torch.Tensor:
    """Per-body L2 error of a kinematic ``quantity`` vs the reference motion, exp-kernelled.

    Shared body of the three jaccard-1.000 motion-tracking rewards (position in the
    anchor-relative frame; linear / angular velocity in world frame). ``quantity``
    selects the ``(reference, robot)`` attribute pair on the :class:`MotionCommand`;
    the error is summed over the spatial axis, averaged over the selected bodies, and
    mapped through ``exp(-error / std^2)``.
    """
    reference_attr, robot_attr = _BODY_ERROR_ATTRS[quantity]
    cmd = _motion_command(env, command_name)
    indexes = _body_index_filter(cmd, body_names)
    reference = getattr(cmd, reference_attr)
    robot = getattr(cmd, robot_attr)
    error = torch.sum((reference[:, indexes] - robot[:, indexes]) ** 2, dim=-1)
    return torch.exp(-error.mean(dim=-1) / (std * std))


def motion_relative_body_position_error_exp(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Multi-body L2 position error against the anchor-aligned reference frames."""
    return motion_body_error_exp(env, command_name, std, body_names, quantity="pos")


def motion_relative_body_orientation_error_exp(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Multi-body geodesic rotation error against the anchor-aligned reference frames."""
    cmd = _motion_command(env, command_name)
    indexes = _body_index_filter(cmd, body_names)
    error = (
        quat_error_magnitude(
            cmd.body_quat_relative_w[:, indexes],
            cmd.robot_body_quat_w[:, indexes],
        )
        ** 2
    )
    return torch.exp(-error.mean(dim=-1) / (std * std))


def motion_global_body_linear_velocity_error_exp(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """L2 world-frame linear-velocity error across the tracked bodies."""
    return motion_body_error_exp(env, command_name, std, body_names, quantity="lin_vel")


def motion_global_body_angular_velocity_error_exp(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """L2 world-frame angular-velocity error across the tracked bodies."""
    return motion_body_error_exp(env, command_name, std, body_names, quantity="ang_vel")
