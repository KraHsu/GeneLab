"""Package-internal helpers shared by the DR event functions."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def resolve_link_indices(env: "ManagerBasedRlEnv", link_names: tuple[str, ...] | None) -> list[int]:
    """Resolve a tuple of link names to integer indices; ``None`` = every link."""
    if link_names is None:
        return list(range(len(env.link_names)))
    return [env.link_names.index(n) for n in link_names]


def normalise_env_ids(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None) -> torch.Tensor:
    """Coerce ``None`` (= "all envs", the startup-mode signal) to a full index tensor."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids
