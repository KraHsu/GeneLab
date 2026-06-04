"""Reorient diagnostic metrics (per-env scalars, averaged over the episode)."""

from typing import TYPE_CHECKING

import torch

from genelab.mdp._helpers import resolve_robot_state
from genelab.utils.math import quat_apply_inverse

from genelab_wuji.reorient.mdp.cage import _cube_pos, _hand_link_ids

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


def fingertip_contact_count(
    env: "EnvContext", sensor_name: str = "hand_object_contact"
) -> torch.Tensor:
    """Number of fingertips in contact with the cube (0-5)."""
    return env.sensors[sensor_name].data.tip_found.float().sum(dim=-1)


def finger_collision_frequency(
    env: "EnvContext", sensor_name: str = "finger_collision"
) -> torch.Tensor:
    """Fraction of steps with at least one finger self-collision."""
    return env.sensors[sensor_name].data.found.float()


def cube_height_above_palm(env: "EnvContext") -> torch.Tensor:
    """Cube height along the palm normal (palm-frame z), m."""
    palm_id, _ = _hand_link_ids(env)
    rs = resolve_robot_state(env, "robot")
    rel = quat_apply_inverse(rs.link_quat_w[:, palm_id], _cube_pos(env) - rs.link_pos[:, palm_id])
    return rel[:, 2]


def success_rate(env: "EnvContext", command_name: str = "reorient_command") -> torch.Tensor:
    """Goals reached so far this episode (proxy for success)."""
    return env.command_manager.get_term(command_name).goal_reach_count.float()  # type: ignore[attr-defined]
