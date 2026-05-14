"""Rich-based rendering helpers for the GeneLab CLI."""

from collections.abc import Iterator
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from genelab.registry import ENVS, ROBOTS, TASKS, Registry, RegistryEntry

RegistryKind = Literal["robots", "envs", "tasks"]


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
        "panel.title": "bold cyan",
        "panel.border": "cyan",
        "table.column": "bold white",
    }
)

console: Console = Console(theme=_theme, highlight=False)
err_console: Console = Console(stderr=True, theme=_theme, highlight=False)


def render_main_help() -> None:
    """Print the landing page shown when ``genelab`` is invoked with no subcommand."""

    robot_count = len(ROBOTS.entries())
    env_count = len(ENVS.entries())
    task_count = len(TASKS.entries())

    console.print()
    console.print("[entry.name]GeneLab[/] — Genesis robot lab CLI")
    console.print()
    console.print("[registry.kind]Quickstart[/]")
    console.print(
        "  [step]genelab cache[/]              "
        "[entry.desc]Create project-local cache directories.[/]"
    )
    console.print("  [step]genelab list tasks[/]         [entry.desc]Show registered tasks.[/]")
    console.print(
        "  [step]genelab info NAME[/]          [entry.desc]Show detail for a registered name.[/]"
    )
    console.print(
        "  [step]genelab play TASK[/]          [entry.desc]Run a task; add --vis for a viewer.[/]"
    )
    console.print(
        "  [step]genelab project new NAME[/]   [entry.desc]Scaffold a new extension package.[/]"
    )
    console.print()
    console.print("[registry.kind]Commands[/]")
    console.print("  [entry.name]Registry[/]    list, info")
    console.print("  [entry.name]Runtime[/]     play, train")
    console.print("  [entry.name]Project[/]     project new")
    console.print("  [entry.name]Utilities[/]   cache")
    console.print()
    console.print(
        f"[entry.detail]Registered:[/] {robot_count} robots, {env_count} envs, {task_count} tasks"
    )
    if robot_count == 0 and env_count == 0 and task_count == 0:
        console.print(
            "[hint]No extensions loaded.[/] Try "
            "[step]genelab --import MODULE ...[/] "
            "or install an extension package."
        )
    else:
        console.print(
            "[entry.detail]Drill in with [step]genelab info NAME[/] for fields and examples.[/]"
        )
    console.print("[entry.detail]Use [step]genelab --help[/] for the full command reference.[/]")
    console.print()


def render_registry(kind: RegistryKind) -> None:
    registry = {"robots": ROBOTS, "envs": ENVS, "tasks": TASKS}[kind]
    console.print(f"Registered [registry.kind]{kind}[/]:")
    entries = registry.entries()
    if not entries:
        console.print("  [entry.detail](none)[/]")
        return
    for entry in entries:
        details = _entry_details(kind, entry.name)
        if entry.examples:
            suffix = f" ({len(entry.examples)} example{'s' if len(entry.examples) != 1 else ''})"
            details = f"{details}{suffix}" if details else suffix.strip()
        console.print(f"  - [entry.name]{entry.name}[/]: [entry.desc]{entry.description}[/]")
        if details:
            console.print(f"      [entry.detail]{details}[/]")


def render_entry_info(name: str) -> None:
    """Render a detail panel for a registered task, env, or robot.

    Looks up ``name`` in the task / env / robot registries in that order. Raises
    ``SystemExit`` with the combined available-name listing when nothing matches.
    """

    candidates: tuple[tuple[str, Registry[object]], ...] = (
        ("task", TASKS),
        ("env", ENVS),
        ("robot", ROBOTS),
    )
    for kind, registry in candidates:
        if name in registry:
            _render_entry_panel(kind, registry.entry(name))
            return
    available = sorted({*TASKS.names(), *ENVS.names(), *ROBOTS.names()})
    listing = ", ".join(available) if available else "<none>"
    raise SystemExit(f"unknown name {name!r}; available: {listing}")


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
    console.print(Panel(body, title="Next steps", border_style="panel.border"))


def render_error(message: str) -> None:
    err_console.print(f"[error]error:[/] {message}")


def iter_overridable_paths(value: object, prefix: str = "") -> Iterator[tuple[str, str, str]]:
    """Yield ``(dotted_path, type_name, default_repr)`` for each dataclass field.

    Walks nested dataclasses depth-first. Fields whose name starts with ``_`` are
    skipped. The runtime type of the current value is used for the type column,
    so opaque ``object``-typed fields (e.g. ``TaskCfg.env``) still surface their
    real downstream dataclass.
    """

    if not is_dataclass(value) or isinstance(value, type):
        return
    for f in fields(value):
        if f.name.startswith("_"):
            continue
        current = getattr(value, f.name)
        path = f.name if not prefix else f"{prefix}.{f.name}"
        if is_dataclass(current) and not isinstance(current, type):
            yield from iter_overridable_paths(current, path)
            continue
        yield (path, _short_type_name(current), _format_default(current))


def _render_entry_panel(kind: str, entry: RegistryEntry[object]) -> None:
    rows: list[RenderableType] = [Text(entry.description, style="entry.desc")]

    instance: object | None = None
    try:
        instance = entry.factory()
    except Exception:  # pragma: no cover — factories with side effects
        instance = None

    if kind == "task" and isinstance(instance, _CfgBacked):
        cfg = instance.cfg
        env_name = getattr(cfg, "env_name", None)
        robot_name = getattr(cfg, "robot_name", None)
        trainable = getattr(cfg, "trainable", None)
        meta_bits: list[str] = []
        if env_name is not None:
            meta_bits.append(f"env=[entry.name]{env_name}[/]")
        if robot_name is not None:
            meta_bits.append(f"robot=[entry.name]{robot_name}[/]")
        if trainable is not None:
            meta_bits.append(f"trainable=[entry.name]{trainable}[/]")
        if meta_bits:
            rows.append(Text(""))
            rows.append(Text.from_markup(", ".join(meta_bits)))

    cfg_value = instance.cfg if isinstance(instance, _CfgBacked) else instance
    if cfg_value is not None and is_dataclass(cfg_value) and not isinstance(cfg_value, type):
        table = _overridable_paths_table(cfg_value)
        if table is not None:
            rows.append(Text(""))
            rows.append(Text("Overridable cfg paths", style="entry.name"))
            rows.append(table)
    elif entry.cfg_type is None:
        rows.append(Text(""))
        rows.append(Text("(no cfg introspection available)", style="entry.detail"))

    if entry.examples:
        rows.append(Text(""))
        rows.append(Text("Examples", style="entry.name"))
        for example in entry.examples:
            rows.append(Text.from_markup(f"  [step]{example}[/]"))

    title = f"[panel.title]{kind.capitalize()}:[/] [entry.name]{entry.name}[/]"
    console.print(Panel(Group(*rows), title=title, border_style="panel.border"))


def _overridable_paths_table(cfg: object) -> Table | None:
    paths = list(iter_overridable_paths(cfg))
    if not paths:
        return None
    table = Table.grid(padding=(0, 2))
    table.add_column("path", style="entry.name")
    table.add_column("type", style="table.column")
    table.add_column("default", style="entry.desc")
    table.add_row("[table.column]Path[/]", "[table.column]Type[/]", "[table.column]Default[/]")
    for dotted, type_name, default_repr in paths:
        table.add_row(dotted, type_name, default_repr)
    return table


def _short_type_name(value: object) -> str:
    if value is None:
        return "None"
    return type(value).__name__


def _format_default(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return "()" if isinstance(value, tuple) else "[]"
        return f"<{type(value).__name__} of {len(value)}>"
    return type(value).__name__


def _entry_details(kind: RegistryKind, name: str) -> str:
    try:
        value = {"robots": ROBOTS, "envs": ENVS, "tasks": TASKS}[kind].get(name)
    except Exception:
        return ""
    if kind == "tasks" and isinstance(value, _CfgBacked):
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
