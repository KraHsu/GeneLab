"""Body-level domain randomization: per-env COM and mass shifts."""

from typing import TYPE_CHECKING

import torch

from genelab.managers.scene_entity_cfg import SceneEntityCfg
from genelab.mdp._helpers import asset_handle
from genelab.mdp.dr._common import normalise_env_ids, resolve_link_indices

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def body_com_offset(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    ranges: dict[int, tuple[float, float]] | None = None,
) -> None:
    """Per-env, per-link offset on the centre-of-mass position.

    Wraps Genesis's :meth:`set_COM_shift` (``rigid_entity.py:4089``). Each axis
    in ``ranges`` is sampled independently; missing axes default to zero offset.
    mjlab's G1 cfg shifts the torso COM by ±2.5 cm in x/y and ±3 cm in z to
    test policy robustness to inaccurate inertial calibration.

    Parameters
    ----------
    ranges
        Dict from axis index (0=x, 1=y, 2=z) to ``(lo, hi)``. Axes absent from
        the dict are filled with zeros. Sampled independently per env per link.
    """
    env_ids = normalise_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    link_indices = resolve_link_indices(env, asset_cfg)
    n_envs = int(env_ids.numel())
    n_links = len(link_indices)
    com_shift = torch.zeros(n_envs, n_links, 3, device=env.device)
    if ranges:
        for axis, (lo, hi) in ranges.items():
            com_shift[..., axis] = torch.empty(n_envs, n_links, device=env.device).uniform_(lo, hi)
    setter = getattr(asset_handle(env, asset_cfg), "set_COM_shift", None)
    if setter is None:
        return
    try:
        setter(com_shift, links_idx_local=link_indices, envs_idx=env_ids)
    except Exception:
        pass


def body_mass_offset(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    ranges: tuple[float, float] = (-0.5, 0.5),
) -> None:
    """Per-env, per-link additive offset on link mass (kg).

    Wraps Genesis's :meth:`set_mass_shift` (``rigid_entity.py:4073``). Not present
    in mjlab's reference G1 cfg, but a common sim2real DR (mass calibration
    error is one of the easier real-world perturbations to model). Mass is
    *added* to the nominal value — Genesis takes a shift, not a multiplier.
    """
    env_ids = normalise_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    link_indices = resolve_link_indices(env, asset_cfg)
    n_envs = int(env_ids.numel())
    n_links = len(link_indices)
    mass_shift = torch.empty(n_envs, n_links, device=env.device).uniform_(*ranges)
    setter = getattr(asset_handle(env, asset_cfg), "set_mass_shift", None)
    if setter is None:
        return
    try:
        setter(mass_shift, links_idx_local=link_indices, envs_idx=env_ids)
    except Exception:
        pass
