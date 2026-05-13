"""Base config dataclass shared by every manager term."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def _noop(*_args: Any, **_kwargs: Any) -> Any:
    return None


@dataclass
class ManagerTermBaseCfg:
    """Config common to observation, reward, termination, command, event, and curriculum terms.

    ``func`` may be a plain callable or a class. Class-style terms are instantiated with
    ``(cfg=term_cfg, env=env)`` by the owning manager so they can cache state at construction.
    Action and command terms use ``class_type`` instead, in which case ``func`` is unused and
    may be left at its no-op default.
    """

    func: Callable[..., Any] = _noop
    params: dict[str, Any] = field(default_factory=dict)
