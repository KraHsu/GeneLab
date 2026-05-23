"""Shared manager base utilities."""

import inspect
from copy import deepcopy
from typing import TYPE_CHECKING, Generic, TypeVar

from genelab.managers.manager_term_cfg import ManagerTermBaseCfg
from genelab.managers.scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from genelab.contracts import EnvContext

TCfg = TypeVar("TCfg", bound=ManagerTermBaseCfg)


def instantiate_class_term(term_cfg: ManagerTermBaseCfg, env: "EnvContext") -> None:
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


class BaseTermManager(Generic[TCfg]):
    """Common base for term-keyed managers (rewards, terminations).

    Pulled out per ADR-0002 / R2.5. Each subclass used to repeat the same
    ``__init__`` body — deepcopy cfg, instantiate parallel
    ``_term_names`` / ``_term_cfgs`` lists, and run
    :func:`instantiate_class_term` per term — before appending its
    domain-specific buffer allocation. Now the registration loop lives
    here and subclasses override :meth:`_post_init` for buffer
    allocation. ``_post_init`` is called at the **end** of the base's
    ``__init__`` so ``self._term_names`` / ``self._term_cfgs`` are
    populated before any subclass buffer touches them — the
    init-order invariant gated by ``tests/test_manager_init_order.py``.

    Generic in ``TCfg`` so each subclass narrows ``cfg`` /
    ``_term_cfgs`` to its concrete term-cfg type (``RewardTermCfg`` /
    ``TerminationTermCfg``) without losing type narrowing at call sites.
    """

    def __init__(
        self,
        cfg: dict[str, TCfg],
        env: "EnvContext",
    ) -> None:
        self._env = env
        self.cfg: dict[str, TCfg] = deepcopy(cfg)
        self._term_names: list[str] = []
        self._term_cfgs: list[TCfg] = []
        for name, term_cfg in self.cfg.items():
            instantiate_class_term(term_cfg, env)
            self._term_names.append(name)
            self._term_cfgs.append(term_cfg)
        self._post_init()

    def _post_init(self) -> None:
        """Override to allocate subclass-specific buffers.

        Runs at the end of :meth:`__init__`, after every term in ``cfg``
        has been registered. Subclasses read ``self._term_names``,
        ``self.num_envs``, ``self.device`` here.
        """

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device

    @property
    def active_terms(self) -> list[str]:
        return list(self._term_names)
