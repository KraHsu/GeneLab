"""Structural contracts (ports) for the domain layer.

Domain terms, managers, and sensors type-hint against these ``Protocol``s instead
of importing the concrete :class:`~genelab.envs.manager_based_rl_env.ManagerBasedRlEnv`
and :class:`~genelab.scene.interactive_scene.InteractiveScene`. This inverts the
pervasive ``domain -> envs`` type coupling: the domain depends on an abstraction,
and a second env/scene implementation becomes possible without touching domain
imports.

This module is **dependency-free** within the genelab bands — it imports nothing
from the domain / rl / cli packages (enforced by import-linter). Complex domain
types are therefore left as ``Any``; the protocol guarantees the *attribute exists*
on the env/scene, while call sites keep their concrete knowledge contextually.

The concrete classes declare conformance via a ``TYPE_CHECKING`` assignment in
their own modules (the adapter side of the port), checked by pyright in CI; a
runtime-free test (``tests/test_contracts.py``) guards against drift.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class SceneContext(Protocol):
    """The scene surface reached through ``env.scene`` — port for ``InteractiveScene``.

    All members are read-only ``@property`` to match ``InteractiveScene`` (a property
    does not satisfy a read-write protocol attribute, and the read-only form is
    covariant so ``dict[str, Articulation]`` satisfies ``dict[str, Any]``).
    """

    @property
    def num_envs(self) -> int: ...
    @property
    def device(self) -> str: ...
    @property
    def env_origins(self) -> torch.Tensor: ...
    @property
    def gs_scene(self) -> Any: ...
    @property
    def terrain(self) -> Any: ...
    @property
    def viewer_closed(self) -> bool: ...
    @property
    def sensors(self) -> dict[str, Any]: ...
    @property
    def recorder_bridge(self) -> Any: ...
    @property
    def articulations(self) -> dict[str, Any]: ...
    @property
    def rigid_objects(self) -> dict[str, Any]: ...


@runtime_checkable
class EnvContext(Protocol):
    """The env surface domain terms / managers use — port for ``ManagerBasedRlEnv``.

    Derived from the attributes the domain actually touches today. Widen
    additively as later redirects need more. Rollout-only methods
    (``step`` / ``reset`` / ``close``) are intentionally excluded — they are not
    term-facing. Members ``ManagerBasedRlEnv`` exposes as read-only ``@property`` are
    declared as properties here; its read-write instance attributes (``cfg``, the
    managers, ``common_step_counter``) stay plain annotations.
    """

    # Read-only properties on ManagerBasedRlEnv.
    @property
    def device(self) -> str: ...
    @property
    def num_envs(self) -> int: ...
    @property
    def dt(self) -> float: ...
    @property
    def physics_dt(self) -> float: ...
    @property
    def max_episode_length(self) -> int: ...
    @property
    def max_episode_length_s(self) -> float: ...
    @property
    def num_actions(self) -> int: ...
    @property
    def episode_length_buf(self) -> torch.Tensor: ...
    @property
    def env_origins(self) -> torch.Tensor: ...
    @property
    def viewer_closed(self) -> bool: ...
    @property
    def scene(self) -> SceneContext: ...
    @property
    def articulations(self) -> dict[str, Any]: ...
    @property
    def sensors(self) -> dict[str, Any]: ...

    # Read-write instance attributes.
    common_step_counter: int
    cfg: Any
    action_manager: Any
    command_manager: Any
    observation_manager: Any
    reward_manager: Any
    termination_manager: Any
    event_manager: Any
    curriculum_manager: Any
    metrics_manager: Any

    def write_joint_state_to_sim(
        self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, env_ids: torch.Tensor
    ) -> None: ...

    def write_root_state_to_sim(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
        root_ang_vel_w: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None: ...


@dataclass
class NoiseCfg(ABC):
    """Base config for additive observation noise.

    Relocated here from ``mdp.noise`` so ``managers`` (the observation manager) can
    hint it without importing ``mdp`` — removing the ``managers → mdp`` back-edge. The
    concrete models (``Unoise`` / ``Gnoise`` / …) live in :mod:`genelab.mdp.noise`,
    which re-exports this base.
    """

    @abstractmethod
    def apply(self, data: torch.Tensor) -> torch.Tensor: ...
