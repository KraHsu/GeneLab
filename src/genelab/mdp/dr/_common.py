"""Package-internal helpers shared by the DR event functions."""

from typing import TYPE_CHECKING

import torch

from genelab.managers.scene_entity_cfg import SceneEntityCfg
from genelab.mdp._helpers import asset_articulation

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


def resolve_link_indices(env: "EnvContext", asset_cfg: SceneEntityCfg) -> list[int]:
    """Pull link indices from a resolved ``SceneEntityCfg``; fall back to every link.

    ``asset_cfg.link_names=None`` is the "no selection" state — for DR functions
    that's the natural "act on every link" semantic (mjlab matches this). The
    manager-level resolve pass runs before this is called, so a present
    ``link_names`` always has matching ``link_ids``. The "all links" count comes from
    ``asset_cfg``'s entity (M3.6).
    """
    if asset_cfg.link_ids is None:
        return list(range(len(asset_articulation(env, asset_cfg).link_names)))
    return list(asset_cfg.link_ids)


def resolve_joint_indices(env: "EnvContext", asset_cfg: SceneEntityCfg) -> list[int]:
    """Pull joint indices from a resolved ``SceneEntityCfg``; fall back to every joint."""
    if asset_cfg.joint_ids is None:
        return list(range(len(asset_articulation(env, asset_cfg).joint_names)))
    return list(asset_cfg.joint_ids)


def normalise_env_ids(env: "EnvContext", env_ids: torch.Tensor | None) -> torch.Tensor:
    """Coerce ``None`` (= "all envs", the startup-mode signal) to a full index tensor."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids
