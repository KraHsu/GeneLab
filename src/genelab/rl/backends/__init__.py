"""RL backend registry.

Backends are keyed by the *type* of the agent config they own. They register
themselves on import; :func:`select_backend` lazily imports the backend modules on
first use so importing ``genelab.rl`` never pulls in an RL library that the user
has not installed (the skrl module guards every skrl import).
"""

from genelab.rl.backends.base import (
    AgentKind,
    Backend,
    PlayContext,
    ProfileArgs,
    TrainContext,
)
from genelab.rl.config import BackendConfig

__all__ = [
    "AgentKind",
    "Backend",
    "PlayContext",
    "ProfileArgs",
    "TrainContext",
    "default_backend",
    "register_backend",
    "select_backend",
]

_BACKENDS: dict[type[BackendConfig], Backend] = {}
_DEFAULT_BACKEND_NAME = "rsl_rl"
_loaded = False


def register_backend(backend: Backend) -> None:
    """Register ``backend`` under its ``cfg_type``. Called by each backend module."""
    _BACKENDS[backend.cfg_type] = backend


def _ensure_loaded() -> None:
    """Import the bundled backend modules once, triggering their registration.

    Every module is safe to import without its RL library installed: rsl_rl's
    wrapper degrades gracefully, and the skrl / sb3 backends guard every
    library-specific import behind a function-local import.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    import importlib

    for name in ("rsl_rl", "skrl", "sb3"):
        importlib.import_module(f"genelab.rl.backends.{name}")


def select_backend(agent_cfg: BackendConfig) -> Backend:
    """Return the backend that owns ``type(agent_cfg)``.

    Raises ``ValueError`` for an agent config no backend has registered.
    """
    _ensure_loaded()
    backend = _BACKENDS.get(type(agent_cfg))
    if backend is None:
        known = ", ".join(sorted(t.__name__ for t in _BACKENDS)) or "<none>"
        raise ValueError(
            f"no RL backend registered for agent cfg type {type(agent_cfg).__name__!r}; "
            f"known agent cfg types: {known}"
        )
    return backend


def default_backend() -> Backend:
    """Return the fallback backend (RSL-RL), used for backend-agnostic zero/random play."""
    _ensure_loaded()
    for backend in _BACKENDS.values():
        if backend.name == _DEFAULT_BACKEND_NAME:
            return backend
    raise RuntimeError(f"default backend {_DEFAULT_BACKEND_NAME!r} is not registered")
