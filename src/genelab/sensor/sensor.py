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
    # Which scene entity the sensor attaches to / reads from (ROADMAP M3.6 / ADR-0012 S4).
    # Defaults to the primary ``"robot"``; multi-robot scenes set it per sensor (e.g.
    # ``"robot_b"``). Resolved via ``genelab.sensor._entity`` with a primary fallback.
    entity_name: str = "robot"

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

    def pre_build_genesis(self, gs_scene: Any, entities: dict[str, Any]) -> None:
        """Pre-``gs_scene.build`` hook: allocate Genesis-side resources that the renderer
        snapshots at build time.

        Called by :class:`~genelab.scene.InteractiveScene` after every entity is spawned
        and before ``gs_scene.build()``. Default: no-op. Sensors whose Genesis resources
        are camera-like (BatchRenderer-snapshotted at build time, e.g.
        :class:`~genelab.sensor.CameraSensor`) override this to register the resource
        early; everything else stays in :meth:`bind` post-build.

        ``entities`` maps entity names to the :class:`~genelab.entity.Articulation` /
        :class:`~genelab.entity.RigidObject` wrappers spawned on ``gs_scene``. Use
        ``entities[self.cfg.entity_name].gs_handle`` to reach the raw Genesis handle when the
        sensor needs to attach to a link (``entity_name`` defaults to the primary ``"robot"``).
        """
        del gs_scene, entities

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
