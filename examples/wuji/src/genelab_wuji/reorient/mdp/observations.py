"""Reorient observation terms (tag-frame cube pose, 6D goal error, state progress)."""

from typing import TYPE_CHECKING

import torch

from genelab.mdp._helpers import resolve_robot_state
from genelab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
)

from genelab_wuji.reorient.constants import TAG_IN_PALM_POS, TAG_IN_PALM_QUAT_WXYZ
from genelab_wuji.reorient.mdp._math import matrix_to_rotation_6d
from genelab_wuji.reorient.mdp._state import cage_counter
from genelab_wuji.reorient.mdp.cage import _hand_link_ids

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


def _cube_pose(env: "EnvContext", object_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    handle = env.scene[object_name].gs_handle  # type: ignore[index]
    pos = torch.as_tensor(handle.get_pos(), device=env.device, dtype=torch.float)
    quat = torch.as_tensor(handle.get_quat(), device=env.device, dtype=torch.float)
    if pos.dim() == 1:
        pos = pos.unsqueeze(0).expand(env.num_envs, -1)
    if quat.dim() == 1:
        quat = quat.unsqueeze(0).expand(env.num_envs, -1)
    return pos, quat


def _tag_pose(env: "EnvContext") -> tuple[torch.Tensor, torch.Tensor]:
    palm_id, _ = _hand_link_ids(env)
    rs = resolve_robot_state(env, "robot")
    palm_pos = rs.link_pos[:, palm_id]
    palm_quat = rs.link_quat_w[:, palm_id]
    tag_in_palm_pos = torch.tensor(TAG_IN_PALM_POS, device=env.device).expand_as(palm_pos)
    tag_in_palm_quat = torch.tensor(TAG_IN_PALM_QUAT_WXYZ, device=env.device).expand_as(palm_quat)
    tag_pos = palm_pos + quat_apply(palm_quat, tag_in_palm_pos)
    tag_quat = quat_mul(palm_quat, tag_in_palm_quat)
    return tag_pos, tag_quat


def cube_pos_in_tag(env: "EnvContext", object_name: str = "object") -> torch.Tensor:
    """Cube position expressed in the wrist tag frame, ``(B, 3)``."""
    cube_pos, _ = _cube_pose(env, object_name)
    tag_pos, tag_quat = _tag_pose(env)
    return quat_apply_inverse(tag_quat, cube_pos - tag_pos)


def goal_rot_err_6d(
    env: "EnvContext", command_name: str = "reorient_command", object_name: str = "object"
) -> torch.Tensor:
    """6D representation of the cube-to-goal relative rotation, ``(B, 6)``."""
    command = env.command_manager.get_term(command_name)
    _, cube_quat = _cube_pose(env, object_name)
    err_quat = quat_mul(cube_quat, quat_conjugate(command.goal_quat))  # type: ignore[attr-defined]
    return matrix_to_rotation_6d(matrix_from_quat(err_quat))


def command_state_progress(
    env: "EnvContext", command_name: str = "reorient_command"
) -> torch.Tensor:
    """``[hold_progress, window_progress, episode_progress, goal_reach_count]``, ``(B, 4)``."""
    command = env.command_manager.get_term(command_name)
    hold = command.hold_counter.float() / max(1, command.cfg.success_hold_steps)  # type: ignore[attr-defined]
    window = command.window_timer.float() / max(1, command.cfg.goal_switch_delay)  # type: ignore[attr-defined]
    episode = env.episode_length_buf.float() / float(env.max_episode_length)
    count = command.goal_reach_count.float()  # type: ignore[attr-defined]
    return torch.stack([hold, window, episode, count], dim=-1)


def cage_counter_progress(env: "EnvContext", max_outside_steps: int = 10) -> torch.Tensor:
    """Soft cage counter normalized to ``[0, 1]``, ``(B,)`` -> unsqueezed to ``(B, 1)``."""
    return (cage_counter(env) / float(max_outside_steps)).clamp(0.0, 1.0)
