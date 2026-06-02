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


def _startup_ids(env: "EnvContext", env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def randomize_cube_physics(
    env: "EnvContext",
    env_ids: torch.Tensor | None,
    object_name: str = "object",
    friction_range: tuple[float, float] = (0.5, 1.5),
    mass_shift_range: tuple[float, float] = (-0.02, 0.02),
) -> None:
    """Per-env startup contact DR on the cube: friction ratio + additive mass shift.

    The cube is a :class:`RigidObject` (single link), not an articulation, so the stock
    ``mdp.dr`` helpers — which resolve articulations and fall back to the robot — can't
    target it; we write through its Genesis handle directly (the same access pattern as
    :func:`reset_object_orientation`). Friction and mass are the contact properties Genesis
    randomizes *per-env* (``set_friction_ratio`` / ``set_mass_shift``)."""
    ids = _startup_ids(env, env_ids)
    if ids.numel() == 0:
        return
    n = int(ids.numel())
    handle = env.scene[object_name].gs_handle  # type: ignore[index]
    friction = torch.empty(n, 1, device=env.device).uniform_(*friction_range)
    mass_shift = torch.empty(n, 1, device=env.device).uniform_(*mass_shift_range)
    for setter, value in (("set_friction_ratio", friction), ("set_mass_shift", mass_shift)):
        fn = getattr(handle, setter, None)
        if fn is None:
            continue
        try:
            fn(value, links_idx_local=[0], envs_idx=ids)
        except Exception:  # noqa: BLE001 - Genesis raises pre-build (unit-test scaffolding)
            pass


def soften_contact_sol_params(
    env: "EnvContext",
    _env_ids: torch.Tensor | None,
    robot_name: str = "robot",
    object_name: str = "object",
    timeconst: float = 0.04,
    dampratio: float = 1.0,
) -> None:
    """Set one slightly-more-compliant constraint ``sol_params`` on the hand collision geoms
    and the cube.

    Genesis stores geom solver params in a global (non-batched) structure, so this is a
    *shared* contact tune, not per-env DR. The nominal Genesis defaults already match
    MuJoCo's ``(timeconst≈0.02, dampratio 1.0)``; softening ``timeconst`` brings the
    effective contact closer to MuJoCo's more compliant behavior and reduces the policy's
    sensitivity to contact stiffness — the dimension the sim2sim transfer is most fragile to.
    ``sol_params`` layout: ``(timeconst, dampratio, dmin, dmax, width, mid, power)``."""
    sol = [timeconst, dampratio, 0.9, 0.95, 1.0e-3, 0.5, 2.0]
    robot = env.scene[robot_name].gs_handle  # type: ignore[index]
    cube = env.scene[object_name].gs_handle  # type: ignore[index]
    for handle in (robot, cube):
        for geom in getattr(handle, "geoms", ()):  # type: ignore[union-attr]
            fn = getattr(geom, "set_sol_params", None)
            if fn is None:
                continue
            try:
                fn(sol)
            except Exception:  # noqa: BLE001 - Genesis raises pre-build (unit-test scaffolding)
                pass


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
