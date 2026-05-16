"""Shared manager base utilities."""

import inspect
from typing import TYPE_CHECKING

from genelab.managers.manager_term_cfg import ManagerTermBaseCfg
from genelab.managers.scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def instantiate_class_term(term_cfg: ManagerTermBaseCfg, env: "ManagerBasedRlEnv") -> None:
    """Resolve ``term_cfg`` against ``env``: any :class:`SceneEntityCfg` values inside
    ``params`` get their ``*_ids`` populated, then class funcs get instantiated.

    Single entrypoint called by every manager's ``__init__`` so per-term setup that
    needs the live env (index resolution, stateful func construction) happens in
    exactly one place. Idempotent re-runs are safe — :meth:`SceneEntityCfg.resolve`
    skips fields whose ids are already filled, and the class-instantiation branch
    only fires when ``func`` is still a class.

    **Order matters**: stateful class terms (e.g. ``feet_swing_height``) cache
    ``asset_cfg.link_ids`` in their ``__init__``. If resolve ran after the class
    instantiation, those reads would see ``None``. So resolve goes first.
    """
    for value in term_cfg.params.values():
        if isinstance(value, SceneEntityCfg):
            value.resolve(env)
    if inspect.isclass(term_cfg.func):
        term_cfg.func = term_cfg.func(cfg=term_cfg, env=env)
