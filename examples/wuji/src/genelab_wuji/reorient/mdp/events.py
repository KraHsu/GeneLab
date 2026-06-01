"""Reset events for the reorient task: randomized cube orientation + cage-state clear."""

from typing import TYPE_CHECKING

import torch

from genelab_wuji.reorient.constants import REORIENT_CUBE_INIT_POS
from genelab_wuji.reorient.mdp._state import cage_counter, disturbance_scale_value
from genelab_wuji.reorient.mdp._math import random_quat

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


def reset_object_orientation(
    env: "EnvContext",
    env_ids: torch.Tensor | None,
    object_name: str = "object",
    pos_noise: float = 0.01,
) -> None:
    """Place the cube at its init position (+ small jitter) with a uniform-random orientation."""
    if env_ids is None or env_ids.numel() == 0:
        return
    n = int(env_ids.numel())
    handle = env.scene[object_name].gs_handle  # type: ignore[index]
    pos = torch.tensor(REORIENT_CUBE_INIT_POS, device=env.device).expand(n, -1).clone()
    if pos_noise > 0:
        pos += torch.empty(n, 3, device=env.device).uniform_(-pos_noise, pos_noise)
    quat = random_quat(n, env.device)
    zeros = torch.zeros(n, 3, device=env.device)
    for setter, value in (
        ("set_pos", pos),
        ("set_quat", quat),
        ("set_vel", zeros),
        ("set_ang", zeros),
    ):
        fn = getattr(handle, setter, None)
        if fn is None:
            continue
        try:
            fn(value, envs_idx=env_ids)
        except TypeError:
            fn(value)


def reset_cage_state(env: "EnvContext", env_ids: torch.Tensor | None) -> None:
    """Zero the soft cage-escape counter for the reset envs."""
    counter = cage_counter(env)
    if env_ids is None:
        counter.zero_()
    else:
        counter[env_ids] = 0.0


def apply_velocity_disturbance(
    env: "EnvContext",
    env_ids: torch.Tensor | None,
    object_name: str = "object",
    min_speed: float = 0.05,
    max_speed: float = 0.15,
) -> None:
    """Add a random-direction linear-velocity impulse to the cube (interval disturbance)."""
    if env_ids is None or env_ids.numel() == 0:
        return
    n = int(env_ids.numel())
    handle = env.scene[object_name].gs_handle  # type: ignore[index]
    cur = torch.as_tensor(handle.get_vel(), device=env.device, dtype=torch.float)
    if cur.dim() == 1:
        cur = cur.unsqueeze(0).expand(env.num_envs, -1)
    direction = torch.randn(n, 3, device=env.device)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    # Ramp the upper speed with the adaptive-episode curriculum scale.
    scale = float(disturbance_scale_value(env)[0])
    effective_max = min_speed + (max_speed - min_speed) * scale
    speed = torch.empty(n, 1, device=env.device).uniform_(min_speed, effective_max)
    new_vel = cur[env_ids] + direction * speed
    set_vel = getattr(handle, "set_vel", None)
    if set_vel is None:
        return
    try:
        set_vel(new_vel, envs_idx=env_ids)
    except TypeError:
        set_vel(new_vel)
