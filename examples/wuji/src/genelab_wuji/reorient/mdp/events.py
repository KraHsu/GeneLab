"""Reset events for the reorient task: randomized cube orientation + cage-state clear."""

from typing import TYPE_CHECKING

import torch

from genelab.entity import write_root_velocity
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
    ):
        fn = getattr(handle, setter, None)
        if fn is None:
            continue
        try:
            fn(value, envs_idx=env_ids)
        except TypeError:
            fn(value)
    write_root_velocity(handle, zeros, zeros, env_ids)


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


def _kicked(
    handle: object,
    getter: str,
    env: "EnvContext",
    env_ids: torch.Tensor,
    n: int,
    lo: float,
    hi: float,
    scale: float,
) -> torch.Tensor | None:
    """Current velocity (via ``getter``) plus a random-direction impulse, for ``env_ids``."""
    fn_get = getattr(handle, getter, None)
    if fn_get is None:
        return None
    cur = torch.as_tensor(fn_get(), device=env.device, dtype=torch.float)
    if cur.dim() == 1:
        cur = cur.unsqueeze(0).expand(env.num_envs, -1)
    direction = torch.randn(n, 3, device=env.device)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    mag = torch.empty(n, 1, device=env.device).uniform_(lo, lo + (hi - lo) * scale)
    return cur[env_ids] + direction * mag


def apply_velocity_disturbance(
    env: "EnvContext",
    env_ids: torch.Tensor | None,
    object_name: str = "object",
    min_speed: float = 0.05,
    max_speed: float = 0.15,
    min_angular: float = 0.5,
    max_angular: float = 2.0,
) -> None:
    """Add a random linear + angular velocity impulse to the cube (interval disturbance).

    The angular kick is the important robustness lever here: it perturbs the cube's rotation
    mid-manipulation, so the policy must keep re-converging from contact-response surprises
    instead of relying on one engine's exact contact feedback. Upper magnitudes ramp with the
    adaptive-episode curriculum scale."""
    if env_ids is None or env_ids.numel() == 0:
        return
    n = int(env_ids.numel())
    handle = env.scene[object_name].gs_handle  # type: ignore[index]
    scale = float(disturbance_scale_value(env)[0])
    vel = _kicked(handle, "get_vel", env, env_ids, n, min_speed, max_speed, scale)
    ang = _kicked(handle, "get_ang", env, env_ids, n, min_angular, max_angular, scale)
    if vel is None or ang is None:
        return
    write_root_velocity(handle, vel, ang, env_ids)
