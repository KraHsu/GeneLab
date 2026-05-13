"""Rich-based rendering helpers for the GeneLab CLI."""

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

from genelab.registry import ENVS, ROBOTS, TASKS

RegistryKind = Literal["robots", "envs", "tasks"]


@runtime_checkable
class _TaskLike(Protocol):
    cfg: object


@runtime_checkable
class _CfgBacked(Protocol):
    cfg: object


_theme = Theme(
    {
        "registry.kind": "bold magenta",
        "entry.name": "bold cyan",
        "entry.desc": "white",
        "entry.detail": "dim",
        "hint": "yellow",
        "error": "bold red",
        "ok": "bold green",
        "step": "cyan",
    }
)

console: Console = Console(theme=_theme, highlight=False, soft_wrap=True)
err_console: Console = Console(stderr=True, theme=_theme, highlight=False, soft_wrap=True)


def render_hint() -> None:
    console.print(
        "Registered GeneLab tasks. Try [hint]genelab list tasks[/] or import an extension package."
    )


def render_registry(kind: RegistryKind) -> None:
    registry = {"robots": ROBOTS, "envs": ENVS, "tasks": TASKS}[kind]
    console.print(f"Registered [registry.kind]{kind}[/]:")
    entries = registry.entries()
    if not entries:
        console.print("  [entry.detail](none)[/]")
        return
    for entry in entries:
        details = _entry_details(kind, entry.name)
        console.print(f"  - [entry.name]{entry.name}[/]: [entry.desc]{entry.description}[/]")
        if details:
            console.print(f"      [entry.detail]{details}[/]")


def render_cache(cache_dir: Path) -> None:
    console.print(f"Using project cache at [ok]{cache_dir}[/]")


def render_project_created(target: Path, task_id: str) -> None:
    body = (
        f"[step]cd[/] {target}\n"
        "[step]uv sync[/]\n"
        "[step]uv run genelab list tasks[/]\n"
        f"[step]uv run genelab play[/] {task_id}"
    )
    console.print(f"Created GeneLab extension project at [ok]{target}[/]")
    console.print(Panel(body, title="Next steps", border_style="entry.detail"))


def render_error(message: str) -> None:
    err_console.print(f"[error]error:[/] {message}")


def _entry_details(kind: RegistryKind, name: str) -> str:
    try:
        value = {"robots": ROBOTS, "envs": ENVS, "tasks": TASKS}[kind].get(name)
    except Exception:
        return ""
    if kind == "tasks" and isinstance(value, _TaskLike):
        cfg = value.cfg
        env_name = getattr(cfg, "env_name", None)
        robot_name = getattr(cfg, "robot_name", None)
        trainable = getattr(cfg, "trainable", None)
        if env_name is not None and robot_name is not None:
            return f"env={env_name}, robot={robot_name}, trainable={trainable}"
    cfg = value.cfg if isinstance(value, _CfgBacked) else None
    if cfg is not None:
        return _short_cfg(cfg)
    return ""


def _short_cfg(cfg: object) -> str:
    if not is_dataclass(cfg) or isinstance(cfg, type):
        return ""
    data = asdict(cfg)
    parts: list[str] = []
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool, Path)) or value is None:
            parts.append(f"{key}={value}")
    return ", ".join(parts[:4])
