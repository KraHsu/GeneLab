"""Core configuration objects and CLI override helpers."""

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, cast, get_args, get_origin, get_type_hints

from genelab.sensor import SensorCfg

type _Annotation = object


@dataclass
class SceneCfg:
    """Common Genesis runtime settings."""

    vis: bool = False
    gpu: bool = False
    steps: int = 240
    dt: float = 0.01
    substeps: int = 4
    num_envs: int = 1
    env_spacing: tuple[float, float] = (2.0, 2.0)
    sensors: tuple[SensorCfg, ...] = field(default_factory=tuple)


@dataclass
class ActionManagerCfg:
    enabled: bool = True


@dataclass
class ObservationManagerCfg:
    enabled: bool = False


@dataclass
class RewardManagerCfg:
    enabled: bool = False


@dataclass
class TerminationManagerCfg:
    enabled: bool = False


@dataclass
class EventManagerCfg:
    enabled: bool = True


@dataclass
class ManagerBasedEnvCfg:
    """Base shape for Isaac Lab-style manager-based environments."""

    scene: SceneCfg = field(default_factory=SceneCfg)
    actions: ActionManagerCfg = field(default_factory=ActionManagerCfg)
    observations: ObservationManagerCfg = field(default_factory=ObservationManagerCfg)
    rewards: RewardManagerCfg = field(default_factory=RewardManagerCfg)
    terminations: TerminationManagerCfg = field(default_factory=TerminationManagerCfg)
    events: EventManagerCfg = field(default_factory=EventManagerCfg)


@dataclass
class TaskCfg:
    """Registry-facing task configuration.

    ``env`` intentionally accepts an arbitrary dataclass so examples and downstream projects can
    plug in their own environment configs without changing the core registry or CLI. ``play_env``
    optionally provides a play-mode variant (curriculum off, push off). ``agent`` carries an
    RL runner cfg (e.g. ``RslRlOnPolicyRunnerCfg``) when the task is trainable.
    """

    name: str
    env_name: str
    robot_name: str
    env: object
    play_env: object | None = None
    agent: object | None = None
    trainable: bool = False


_PATH_ALIASES = {
    "vis": "env.scene.vis",
    "gpu": "env.scene.gpu",
    "steps": "env.scene.steps",
    "dt": "env.scene.dt",
}


def apply_overrides(cfg: object, overrides: dict[str, str]) -> None:
    """Apply dotted CLI overrides to a dataclass config object."""

    for raw_path, raw_value in overrides.items():
        path = _PATH_ALIASES.get(raw_path, raw_path)
        _set_dotted_value(cfg, path, raw_value)


def _set_dotted_value(root: object, dotted_path: str, raw_value: str) -> None:
    parts = dotted_path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid override path: {dotted_path!r}")
    target = root
    for part in parts[:-1]:
        target = _descend(target, part, dotted_path)
    field_name = parts[-1]
    if isinstance(target, dict):
        if field_name not in target:
            raise ValueError(f"unknown override path: {dotted_path!r}")
        current = target[field_name]
        annotation = None
        target[field_name] = _coerce_value(raw_value, current, annotation)
        return
    if not hasattr(target, field_name):
        raise ValueError(f"unknown override path: {dotted_path!r}")
    current = getattr(target, field_name)
    annotation = _field_annotation(target, field_name)
    setattr(target, field_name, _coerce_value(raw_value, current, annotation))


def _descend(target: object, key: str, dotted_path: str) -> object:
    if isinstance(target, dict):
        if key not in target:
            raise ValueError(f"unknown override path: {dotted_path!r}")
        return target[key]
    if not hasattr(target, key):
        raise ValueError(f"unknown override path: {dotted_path!r}")
    return getattr(target, key)


def _field_annotation(obj: object, field_name: str) -> _Annotation | None:
    if not is_dataclass(obj):
        return None
    return cast(_Annotation | None, get_type_hints(type(obj)).get(field_name))


def _coerce_value(raw_value: str, current: object, annotation: _Annotation | None) -> object:
    if raw_value.lower() in {"none", "null"}:
        return None
    target_type = _strip_optional(annotation)
    if target_type in (None, Any):
        target_type = type(current) if current is not None else str
    origin = get_origin(target_type)
    if target_type is bool or isinstance(current, bool):
        return _parse_bool(raw_value)
    if target_type is int or isinstance(current, int) and not isinstance(current, bool):
        return int(raw_value)
    if target_type is float or isinstance(current, float):
        return float(raw_value)
    if target_type is Path or isinstance(current, Path):
        return Path(raw_value)
    if origin in (tuple, list):
        values = [part.strip() for part in raw_value.split(",") if part.strip()]
        item_types = get_args(target_type)
        if origin is tuple and item_types:
            if len(item_types) == 2 and item_types[1] is Ellipsis:
                coerced = tuple(_coerce_scalar(value, item_types[0]) for value in values)
            else:
                coerced = tuple(
                    _coerce_scalar(value, typ)
                    for value, typ in zip(values, item_types, strict=True)
                )
            return coerced
        item_type = item_types[0] if item_types else str
        return [_coerce_scalar(value, item_type) for value in values]
    return _coerce_scalar(raw_value, target_type)


def _strip_optional(annotation: _Annotation | None) -> _Annotation | None:
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    if origin is not UnionType:
        return annotation
    args = tuple(arg for arg in get_args(annotation) if arg is not type(None))
    if len(args) == 1 and len(args) != len(get_args(annotation)):
        return args[0]
    return annotation


def _coerce_scalar(raw_value: str, target_type: _Annotation | None) -> object:
    if target_type is bool:
        return _parse_bool(raw_value)
    if target_type is int:
        return int(raw_value)
    if target_type is float:
        return float(raw_value)
    if target_type is Path:
        return Path(raw_value)
    return raw_value


def _parse_bool(raw_value: str) -> bool:
    value = raw_value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"expected a boolean value, got {raw_value!r}")
