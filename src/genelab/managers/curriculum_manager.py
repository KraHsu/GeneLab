"""Curriculum manager: per-term hooks called on reset to mutate env config."""

from dataclasses import dataclass

import torch

from genelab.managers._base import BaseTermManager
from genelab.managers.manager_term_cfg import ManagerTermBaseCfg


@dataclass
class CurriculumTermCfg(ManagerTermBaseCfg):
    pass


class CurriculumManager(BaseTermManager[CurriculumTermCfg]):
    def compute(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
        extras: dict[str, float] = {}
        # Collect tensor-valued curriculum terms separately so all their means come back
        # with a single host sync (one ``.tolist()`` instead of one ``.item()`` per term).
        tensor_keys: list[str] = []
        tensor_means: list[torch.Tensor] = []
        for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=True):
            value = term_cfg.func(self._env, env_ids, **term_cfg.params)
            if isinstance(value, torch.Tensor):
                tensor_keys.append(f"Curriculum/{name}")
                tensor_means.append(value.float().mean())
            elif isinstance(value, dict):
                # Multi-axis curriculum terms (e.g. ``mdp.commands_vel``) return a dict of
                # named tensors; expand each entry under ``Curriculum/<term>/<key>`` and
                # pull them through the same single-sync host transfer as scalar terms.
                for sub_name, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        tensor_keys.append(f"Curriculum/{name}/{sub_name}")
                        tensor_means.append(sub_value.float().mean())
                    elif sub_value is not None:
                        extras[f"Curriculum/{name}/{sub_name}"] = float(sub_value)
            elif value is not None:
                extras[f"Curriculum/{name}"] = float(value)
        if tensor_means:
            means_list = torch.stack(tensor_means).tolist()
            for key, mean in zip(tensor_keys, means_list):
                extras[key] = float(mean)
        return extras
