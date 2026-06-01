"""Reorient reward terms (task + grasp regularization).

Contact-driven terms (tip_slide, palm_detach, finger_collision) live alongside the custom
contact sensor; this module covers the orientation task reward, the escalating hold bonus,
and the hand-pose regularizer. Effort / action-rate penalties reuse ``genelab.mdp``.
"""

import math
import re
from typing import TYPE_CHECKING

import torch

from genelab.mdp._helpers import resolve_articulation, resolve_robot_state

if TYPE_CHECKING:
    from genelab.contracts import EnvContext

_TIP_LINK4_RE = re.compile(r"right_finger[1-5]_link4$")


def _tip_link4_ids(env: "EnvContext") -> torch.Tensor:
    cached = getattr(env, "_wuji_tip_link4_ids", None)
    if cached is not None:
        return cached
    names = resolve_articulation(env, "robot").link_names
    ids = torch.tensor(
        [i for i, n in enumerate(names) if _TIP_LINK4_RE.search(n)], device=env.device
    )
    env._wuji_tip_link4_ids = ids  # type: ignore[attr-defined]
    return ids


def _cube_lin_vel(env: "EnvContext", object_name: str = "object") -> torch.Tensor:
    handle = env.scene[object_name].gs_handle  # type: ignore[index]
    vel = torch.as_tensor(handle.get_vel(), device=env.device, dtype=torch.float)
    return vel.unsqueeze(0).expand(env.num_envs, -1) if vel.dim() == 1 else vel


def orientation_alignment(
    env: "EnvContext",
    command_name: str = "reorient_command",
    bound: float = 0.2,
    margin: float = math.pi,
) -> torch.Tensor:
    """Linear-tolerance reward on the cube-to-goal geodesic error (1 inside ``bound``)."""
    command = env.command_manager.get_term(command_name)
    err = command.ori_error  # type: ignore[attr-defined]
    return (1.0 - (err - bound).clamp(min=0.0) / margin).clamp(min=0.0)


def hold_escalation(env: "EnvContext", command_name: str = "reorient_command") -> torch.Tensor:
    """Time-escalating bonus: the hold-counter value while in the success window + on target."""
    command = env.command_manager.get_term(command_name)
    active = command.in_success_window & command.within_threshold  # type: ignore[attr-defined]
    return active.float() * command.reward_hold_counter.float()  # type: ignore[attr-defined]


def hand_pose_penalty(env: "EnvContext") -> torch.Tensor:
    """Sum of squared deviations of the finger joints from the home keyframe."""
    rs = resolve_robot_state(env, "robot")
    default = resolve_articulation(env, "robot").default_joint_pos
    return torch.sum((rs.joint_pos - default.unsqueeze(0)) ** 2, dim=-1)


def tip_slide_penalty(env: "EnvContext", sensor_name: str = "hand_object_contact") -> torch.Tensor:
    """Penalize fingertip sliding: tip-vs-cube relative speed summed over contacting tips."""
    data = env.sensors[sensor_name].data
    rs = resolve_robot_state(env, "robot")
    tip_vel = rs.link_lin_vel_w[:, _tip_link4_ids(env)]  # (B, 5, 3)
    rel = (tip_vel - _cube_lin_vel(env).unsqueeze(1)).norm(dim=-1)  # (B, 5)
    return torch.sum(rel * data.tip_found.float(), dim=-1)


def palm_detach_reward(env: "EnvContext", sensor_name: str = "hand_object_contact") -> torch.Tensor:
    """Reward holding the cube on the distal fingers, clear of the palm/proximal links."""
    data = env.sensors[sensor_name].data
    return ((~data.palm_found) & data.distal_found).float()


def finger_self_collision_penalty(
    env: "EnvContext", sensor_name: str = "finger_collision"
) -> torch.Tensor:
    """Penalize finger self-collision (any self-contact pair above the sensor threshold)."""
    return env.sensors[sensor_name].data.found.float()
