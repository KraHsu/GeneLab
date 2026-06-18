"""Per-environment gravity-direction domain randomization."""

import math
from typing import TYPE_CHECKING

import torch

from genelab.mdp.dr._common import normalise_env_ids

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


def _rigid_solver(env: "EnvContext"):
    """Best-effort access to the Genesis rigid solver (per-env gravity lives there)."""
    scene = getattr(env, "scene", None)
    gs_scene = getattr(scene, "_gs_scene", None)
    sim = getattr(gs_scene, "sim", None)
    return getattr(sim, "rigid_solver", None)


def gravity_tilt(
    env: "EnvContext",
    env_ids: torch.Tensor | None,
    max_tilt_rad: float = 0.4,
    magnitude: float = 9.81,
) -> None:
    """Per-env gravity-direction DR: tilt gravity by a random polar angle in a random azimuth.

    The Genesis-native equivalent of randomizing a fixed-base hand's mount pitch (mjlab's
    ``reset_root_state`` pitch DR). Genesis refuses per-env orientation on a fixed-base link
    with geometry, but tilting **gravity** per env gives the SAME gravity-in-palm physics
    while keeping the hand fixed-base and the wrist-tag world frame (``tag_w``) unchanged —
    so the deploy obs pipeline needs no frame changes. Makes the policy robust to a tilted
    hardware mount (e.g. a ~10 deg down-tilt). A full cone (random azimuth) covers the mount
    tilt regardless of axis. Use ``mode="reset"`` (re-sample per episode, like mjlab).

    Per-env gravity requires no batch flag — gravity is already a per-env solver field
    (``rigid_solver.set_gravity(..., envs_idx=...)``). Guarded for the fake-env test scaffold.
    """
    env_ids = normalise_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    setter = getattr(_rigid_solver(env), "set_gravity", None)
    if setter is None:
        return
    n = int(env_ids.numel())
    theta = torch.empty(n, device=env.device).uniform_(0.0, max_tilt_rad)
    phi = torch.empty(n, device=env.device).uniform_(0.0, 2.0 * math.pi)
    horizontal = magnitude * torch.sin(theta)
    g = torch.empty(n, 3, device=env.device)
    g[:, 0] = horizontal * torch.cos(phi)
    g[:, 1] = horizontal * torch.sin(phi)
    g[:, 2] = -magnitude * torch.cos(theta)
    try:
        setter(g, envs_idx=env_ids)
    except Exception:
        pass
