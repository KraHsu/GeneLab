"""Small registries for robots, environments, tasks, and downstream extensions."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import importlib
from importlib import metadata
from pathlib import Path
import sys
from typing import Any, Final, cast


@dataclass(frozen=True)
class RegistryEntry[T]:
    """A named factory/config entry exposed to the CLI."""

    name: str
    description: str
    factory: Callable[[], T]
    cfg_type: type[object] | None = None


class Registry[T]:
    """Map stable names to lazily-created registry values."""

    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, RegistryEntry[T]] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], T],
        *,
        description: str,
        cfg_type: type[object] | None = None,
    ) -> RegistryEntry[T]:
        if name in self._entries:
            raise ValueError(f"{self.kind} already registered: {name}")
        entry = RegistryEntry(
            name=name, description=description, factory=factory, cfg_type=cfg_type
        )
        self._entries[name] = entry
        return entry

    def get(self, name: str) -> T:
        try:
            return self._entries[name].factory()
        except KeyError as exc:
            choices = ", ".join(self.names()) or "<none>"
            raise KeyError(f"unknown {self.kind} {name!r}; available: {choices}") from exc

    def entry(self, name: str) -> RegistryEntry[T]:
        try:
            return self._entries[name]
        except KeyError as exc:
            choices = ", ".join(self.names()) or "<none>"
            raise KeyError(f"unknown {self.kind} {name!r}; available: {choices}") from exc

    def names(self) -> list[str]:
        return sorted(self._entries)

    def entries(self) -> list[RegistryEntry[T]]:
        return [self._entries[name] for name in self.names()]

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterable[RegistryEntry[T]]:
        return iter(self.entries())


ROBOTS: Registry[Any] = Registry("robot")
ENVS: Registry[Any] = Registry("env")
TASKS: Registry[Any] = Registry("task")
ENTRYPOINT_GROUP: Final = "genelab.extensions"
_loaded_extension_modules: set[str] = set()
_loaded_entrypoints: set[str] = set()


def register_robot[T](
    name: str,
    factory: Callable[[], T],
    *,
    description: str,
    cfg_type: type[object] | None = None,
) -> RegistryEntry[T]:
    return ROBOTS.register(name, factory, description=description, cfg_type=cfg_type)


def register_env[T](
    name: str,
    factory: Callable[[], T],
    *,
    description: str,
    cfg_type: type[object] | None = None,
) -> RegistryEntry[T]:
    return ENVS.register(name, factory, description=description, cfg_type=cfg_type)


def register_task[T](
    name: str,
    factory: Callable[[], T],
    *,
    description: str,
    cfg_type: type[object] | None = None,
) -> RegistryEntry[T]:
    return TASKS.register(name, factory, description=description, cfg_type=cfg_type)


def load_builtin_registries() -> None:
    """Compatibility no-op.

    Example tasks live in normal extension packages and are loaded through entry points or
    ``load_extension_module``.
    """


def load_extension_module(module_name: str) -> None:
    """Load a downstream extension module and call its ``register()`` hook if present.

    External projects can either register objects as an import side effect or expose a no-argument
    ``register`` function. The module name is recorded so repeated CLI invocations in one process do
    not double-register the same extension.
    """

    if module_name in _loaded_extension_modules:
        return
    _ensure_cwd_on_sys_path()
    module = importlib.import_module(module_name)
    register_hook = getattr(module, "register", None)
    if register_hook is not None:
        if not callable(register_hook):
            raise TypeError(f"extension module {module_name!r} defines non-callable register")
        register_hook()
    _loaded_extension_modules.add(module_name)


def _ensure_cwd_on_sys_path() -> None:
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def load_entrypoint_extensions(group: str = ENTRYPOINT_GROUP) -> None:
    """Load installed GeneLab extension entry points.

    Entry points should target a no-argument registration function, for example
    ``my_robot_project.tasks:register``. An entry point that resolves to a module is also accepted
    when that module defines a callable ``register`` hook.
    """

    for entry_point in metadata.entry_points(group=group):
        key = f"{group}:{entry_point.name}={entry_point.value}"
        module_name = cast(str, getattr(entry_point, "module", ""))
        if key in _loaded_entrypoints:
            continue
        if module_name in _loaded_extension_modules:
            _loaded_entrypoints.add(key)
            continue
        loaded = entry_point.load()
        if callable(loaded):
            loaded()
        else:
            register_hook = getattr(loaded, "register", None)
            if not callable(register_hook):
                raise TypeError(
                    f"GeneLab extension entry point {entry_point.name!r} must point to a callable "
                    "or to a module with callable register()"
                )
            register_hook()
        if module_name:
            _loaded_extension_modules.add(module_name)
        _loaded_entrypoints.add(key)
