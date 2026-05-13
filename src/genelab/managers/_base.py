"""Shared manager base utilities."""

import inspect
from typing import TYPE_CHECKING

from genelab.managers.manager_term_cfg import ManagerTermBaseCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def instantiate_class_term(term_cfg: ManagerTermBaseCfg, env: "ManagerBasedRlEnv") -> None:
    """If ``term_cfg.func`` is a class, replace it with an instance bound to ``env``."""
    if inspect.isclass(term_cfg.func):
        term_cfg.func = term_cfg.func(cfg=term_cfg, env=env)
