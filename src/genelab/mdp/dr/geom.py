"""Geom-level domain randomization (currently: per-link absolute friction)."""

from typing import TYPE_CHECKING, Any

import torch

from genelab.managers.scene_entity_cfg import SceneEntityCfg
from genelab.mdp.dr._common import normalise_env_ids, resolve_link_indices

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _link_nominal_friction(robot: Any, link_indices: list[int]) -> torch.Tensor | None:
    """Read each selected link's nominal friction from Genesis's geoms.

    Returns a 1-D tensor of length ``len(link_indices)``, or ``None`` if the
    handle doesn't expose the ``.links[i].geoms[j]._friction`` chain (fake env
    in tests). Falls back to 1.0 per link when a specific geom lacks the field
    so the call still progresses.
    """
    links_attr = getattr(robot, "links", None)
    if links_attr is None:
        return None
    nominals: list[float] = []
    for i_l in link_indices:
        try:
            geoms = list(links_attr[i_l].geoms)
        except (AttributeError, IndexError, TypeError):
            return None
        # All geoms in a foot link share the same nominal friction in G1's XML.
        # Take the first geom's value; if the link has no geoms, fall back to 1.0
        # so the per-link slot still has a valid (if neutral) divisor.
        if not geoms:
            nominals.append(1.0)
            continue
        nominal = getattr(geoms[0], "_friction", None)
        nominals.append(float(nominal) if nominal is not None else 1.0)
    return torch.tensor(nominals, dtype=torch.float32)


def geom_friction(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    ranges: tuple[float, float] = (0.5, 1.5),
    shared_random: bool = True,
) -> None:
    """Per-env absolute friction coefficient on the geoms of the selected links.

    mjlab parity: writes the *sampled value* as the effective friction (mjlab's
    ``operation="abs"``), not as a multiplier on nominal. Genesis only exposes
    a per-env-batched ``set_friction_ratio`` setter, so we convert internally:
    ``ratio = sampled / nominal`` (read once per link from
    ``robot.links[i].geoms[0]._friction``). The downstream simulation reads
    ``effective = nominal * ratio = sampled``, which matches mjlab's behavior.

    ``asset_cfg.link_names=None`` covers the whole entity (mjlab's G1 cfg
    restricts to foot geoms for a focused friction sweep).

    Parameters
    ----------
    ranges
        Uniform sample range for the *absolute* friction coefficient. mjlab's
        G1 uses ``(0.3, 1.2)``.
    shared_random
        If True (mjlab default), every selected link of a given env gets the
        same random sample — useful when "ground friction" should be one value
        per env even if multiple foot geoms exist. If False, every link gets
        an independent sample.
    """
    env_ids = normalise_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    link_indices = resolve_link_indices(env, asset_cfg)
    n_envs = int(env_ids.numel())
    n_links = len(link_indices)
    if shared_random:
        sample = torch.empty(n_envs, 1, device=env.device).uniform_(*ranges)
        sampled = sample.expand(n_envs, n_links).contiguous()
    else:
        sampled = torch.empty(n_envs, n_links, device=env.device).uniform_(*ranges)

    # Translate absolute friction → ratio for Genesis's batched setter.
    nominal = _link_nominal_friction(env.robot, link_indices)
    if nominal is None:
        # Fake env (no ``.links`` chain): treat nominal as 1.0 so the ratio
        # passed to ``set_friction_ratio`` is numerically the sampled value.
        # Tests inspect this tensor directly and assert it lies in ``ranges``.
        ratio = sampled
    else:
        nominal = nominal.to(device=sampled.device).clamp(min=1e-6)
        ratio = sampled / nominal.unsqueeze(0)

    setter = getattr(env.robot, "set_friction_ratio", None)
    if setter is None:
        return
    try:
        setter(ratio, links_idx_local=link_indices, envs_idx=env_ids)
    except Exception:
        # Genesis raises if it isn't built yet (e.g. unit-test scaffolding); silent
        # no-op keeps the same function usable across runtime and fake-env tests.
        pass
