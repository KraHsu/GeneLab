"""Backend-agnostic sensor abstraction.

The shape mirrors mjlab's ``SensorCfg`` / ``Sensor[T]`` so configs and observation terms can move
between the two backends without ceremony, but the lifecycle drops mjlab's ``edit_spec`` /
``initialize`` pair — Genesis has no MJCF spec to rewrite, so ``bind(env)`` is the single hook.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class SensorCfg(ABC):
    """Backend-agnostic sensor configuration. Subclasses describe what to sense and how."""

    name: str = ""

    @abstractmethod
    def build(self) -> "Sensor[Any]": ...


class Sensor[T](ABC):
    """Per-step cached sensor. Subclasses implement ``_compute_data``.

    Lifecycle: ``__init__(cfg)`` → ``bind(env)`` once → per-step ``update(dt)`` (invalidates the
    cache) → ``data`` triggers a lazy ``_compute_data`` on first access → ``reset(env_ids)``
    invalidates and lets subclasses clear stateful buffers.
    """

    def __init__(self, cfg: SensorCfg) -> None:
        self._cfg = cfg
        self._env: "ManagerBasedRlEnv | None" = None
        self._cached_data: T | None = None
        self._cache_valid: bool = False

    @property
    def cfg(self) -> SensorCfg:
        return self._cfg

    @property
    def name(self) -> str:
        return self._cfg.name

    def bind(self, env: "ManagerBasedRlEnv") -> None:
        """Called once during env construction. Subclasses may cache link indices etc."""
        self._env = env

    @property
    def data(self) -> T:
        if not self._cache_valid:
            self._cached_data = self._compute_data()
            self._cache_valid = True
        return cast(T, self._cached_data)

    def _invalidate_cache(self) -> None:
        self._cache_valid = False

    def update(self, dt: float) -> None:
        """Called each control step after ``_refresh_robot_state``. Override + ``super().update(dt)``."""
        del dt
        self._invalidate_cache()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Called per-env on reset. Override + ``super().reset(env_ids)`` to clear state buffers."""
        del env_ids
        self._invalidate_cache()

    @abstractmethod
    def _compute_data(self) -> T: ...
