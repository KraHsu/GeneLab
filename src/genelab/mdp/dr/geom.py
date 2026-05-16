"""Geom-level domain randomization (currently: per-link friction ratio)."""

from typing import TYPE_CHECKING

import torch

from genelab.mdp.dr._common import normalise_env_ids, resolve_link_indices

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def geom_friction(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    link_names: tuple[str, ...] | None = None,
    ranges: tuple[float, float] = (0.5, 1.5),
    shared_random: bool = True,
) -> None:
    """Per-env friction-coefficient multiplier on the geoms of the selected links.

    Calls Genesis's :meth:`set_friction_ratio` (``rigid_entity.py:4018``), which
    multiplies the configured friction of every geom owned by the listed links
    by the sampled ratio. ``link_names=None`` covers the whole entity (mjlab's
    G1 cfg restricts to foot geoms for a focused friction sweep).

    Parameters
    ----------
    ranges
        Uniform sample range for the multiplier. Default ``(0.5, 1.5)`` centres
        on the configured nominal; mjlab's G1 uses ``(0.3, 1.2)``.
    shared_random
        If True (mjlab default), every selected link of a given env shares the
        same random sample — useful when "ground friction" should be one value
        per env even if multiple foot geoms exist. If False, every link gets
        an independent sample.
    """
    env_ids = normalise_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    link_indices = resolve_link_indices(env, link_names)
    n_envs = int(env_ids.numel())
    n_links = len(link_indices)
    if shared_random:
        sample = torch.empty(n_envs, 1, device=env.device).uniform_(*ranges)
        friction = sample.expand(n_envs, n_links).contiguous()
    else:
        friction = torch.empty(n_envs, n_links, device=env.device).uniform_(*ranges)
    setter = getattr(env.robot, "set_friction_ratio", None)
    if setter is None:
        return
    try:
        setter(friction, links_idx_local=link_indices, envs_idx=env_ids)
    except Exception:
        # Genesis raises if it isn't built yet (e.g. unit-test scaffolding); silent
        # no-op keeps the same function usable across runtime and fake-env tests.
        pass
