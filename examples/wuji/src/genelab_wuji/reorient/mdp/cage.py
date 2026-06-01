"""Palm-relative AABB "cage": escape distance, soft counter penalty, drop termination.

The cage is the axis-aligned bounding box of the hand links expressed in the palm frame
(padded by ``margin``, with extra headroom above). The cube's signed escape distance feeds
a soft counter that grows while the cube is outside and decays while inside; a large counter
penalizes escape and ultimately terminates the episode (cube dropped).
"""

import re
from typing import TYPE_CHECKING

import torch

from genelab.mdp._helpers import resolve_articulation, resolve_robot_state
from genelab.utils.math import quat_apply_inverse

from genelab_wuji.reorient.constants import PALM_BODY_NAME
from genelab_wuji.reorient.mdp._state import cage_counter

if TYPE_CHECKING:
    from genelab.contracts import EnvContext

_HAND_LINK_RE = re.compile(r"(.*_palm_link|.*_finger.*_link[1-4])$")


def _hand_link_ids(env: "EnvContext") -> tuple[int, list[int]]:
    cached = getattr(env, "_wuji_hand_link_ids", None)
    if cached is not None:
        return cached
    robot = resolve_articulation(env, "robot")
    names = robot.link_names
    palm_id = names.index(PALM_BODY_NAME)
    hand_ids = [i for i, n in enumerate(names) if _HAND_LINK_RE.match(n)]
    result = (palm_id, hand_ids)
    env._wuji_hand_link_ids = result  # type: ignore[attr-defined]
    return result


def _cube_pos(env: "EnvContext", object_name: str = "object") -> torch.Tensor:
    handle = env.scene[object_name].gs_handle  # type: ignore[index]
    pos = torch.as_tensor(handle.get_pos(), device=env.device, dtype=torch.float)
    return pos.unsqueeze(0).expand(env.num_envs, -1) if pos.dim() == 1 else pos


def _escape_distance(env: "EnvContext", margin: float, z_upper_margin: float) -> torch.Tensor:
    """Sum of per-axis AABB escape distances of the cube in the palm frame (0 if inside)."""
    palm_id, hand_ids = _hand_link_ids(env)
    rs = resolve_robot_state(env, "robot")
    palm_pos = rs.link_pos[:, palm_id]  # (B, 3)
    palm_quat = rs.link_quat_w[:, palm_id]  # (B, 4)
    hand_pos = rs.link_pos[:, hand_ids]  # (B, H, 3)
    h = hand_pos.shape[1]
    rel = quat_apply_inverse(
        palm_quat.unsqueeze(1).expand(-1, h, -1), hand_pos - palm_pos.unsqueeze(1)
    )  # (B, H, 3)
    lo = rel.amin(dim=1)  # (B, 3)
    hi = rel.amax(dim=1)
    xy_min, z_min = lo[:, :2] - margin, lo[:, 2] - margin
    xy_max, z_max = hi[:, :2] + margin, hi[:, 2] + z_upper_margin

    cube_rel = quat_apply_inverse(palm_quat, _cube_pos(env) - palm_pos)  # (B, 3)
    d_xy = (xy_min - cube_rel[:, :2]).clamp(min=0.0) + (cube_rel[:, :2] - xy_max).clamp(min=0.0)
    d_z = (z_min - cube_rel[:, 2]).clamp(min=0.0) + (cube_rel[:, 2] - z_max).clamp(min=0.0)
    return d_xy.sum(dim=-1) + d_z


def update_cage_counter(
    env: "EnvContext",
    margin: float = 0.01,
    z_upper_margin: float = 0.03,
    counter_decay: float = 0.5,
    counter_cap: float = 15.0,
) -> torch.Tensor:
    """Advance and return the soft escape counter (grows outside, decays inside)."""
    outside = _escape_distance(env, margin, z_upper_margin) > 0.0
    counter = cage_counter(env)
    step = torch.where(outside, torch.ones_like(counter), torch.full_like(counter, -counter_decay))
    counter.copy_((counter + step).clamp_(0.0, counter_cap))
    return outside.float()


def cage_escape_penalty(
    env: "EnvContext",
    margin: float = 0.01,
    z_upper_margin: float = 0.03,
    drop_scale: float = 4.0,
) -> torch.Tensor:
    """Escape penalty magnitude (>= 0); updates the shared counter. Apply a negative weight."""
    outside = update_cage_counter(env, margin=margin, z_upper_margin=z_upper_margin)
    return outside * (cage_counter(env) / 10.0) * drop_scale


def cage_drop(env: "EnvContext", max_outside_steps: int = 10) -> torch.Tensor:
    """Terminate when the cube has been outside the cage long enough (counter saturates)."""
    return cage_counter(env) >= float(max_outside_steps)
