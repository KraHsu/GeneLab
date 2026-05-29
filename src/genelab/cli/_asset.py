"""``genelab asset`` — list / info / download / purge for the asset zoo.

Walks ``genelab.asset_zoo``'s submodules to enumerate every module-level
:class:`~genelab.utils.download.AssetSpec` declared there (the existing factories
already pin them as module-private ``_MJCF`` / similar constants), so this command
group needs no registration in the asset modules themselves. Downloads go through
the same :func:`fetch_progress` Rich callback every other CLI surface uses, so the
progress bar renders the same way whether a user explicitly calls ``genelab asset
download go1`` or implicitly triggers a fetch via ``genelab info go1``.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from genelab.cli._progress import fetch_progress
from genelab.cli._render import console, err_console
from genelab.utils.download import AssetDownloadError, AssetSpec, fetch_asset
from genelab.utils.paths import CACHE_DIR

asset_app: typer.Typer = typer.Typer(
    name="asset",
    help="List, inspect, download, and purge GeneLab asset-zoo files.",
    no_args_is_help=True,
)

_ASSET_ROOT: Path = CACHE_DIR / "assets"


@dataclass(frozen=True)
class _ZooAsset:
    """A discovered ``AssetSpec`` plus where it was found."""

    spec: AssetSpec
    module: str
    attr: str


def _discover_assets() -> list[_ZooAsset]:
    """Walk ``genelab.asset_zoo`` and return every module-level ``AssetSpec`` instance.

    Each submodule is imported once (the package's existing import-time registration
    is idempotent). Module-level constants are detected by ``isinstance(value,
    AssetSpec)``; underscore-prefixed names (``_MJCF`` etc.) are included since that's
    the established convention.
    """
    import genelab.asset_zoo as zoo

    found: list[_ZooAsset] = []
    seen: set[tuple[str, str]] = set()  # (name, md5) dedupe — same asset reused across modules
    for mod_info in pkgutil.iter_modules(zoo.__path__):
        full = f"{zoo.__name__}.{mod_info.name}"
        module = importlib.import_module(full)
        for attr_name, value in inspect.getmembers(module):
            if not isinstance(value, AssetSpec):
                continue
            key = (value.name, value.md5)
            if key in seen:
                continue
            seen.add(key)
            found.append(_ZooAsset(spec=value, module=full, attr=attr_name))
    return sorted(found, key=lambda a: a.spec.name)


def _cache_paths(spec: AssetSpec) -> tuple[Path, Path | None]:
    """Return ``(per-asset cache root, expected entry path)`` for ``spec``.

    The root is ``<CACHE_DIR>/assets/<name>/<md5>``. The entry path is the file
    :func:`fetch_asset` returns: ``<root>/<filename>`` for single-file mode,
    ``<root>/extracted/<archive_member>`` for archive mode (or ``None`` for spec-
    introspection callers).
    """
    root = _ASSET_ROOT / spec.name / spec.md5
    if spec.archive_member is None:
        return root, root / spec.filename
    return root, root / "extracted" / spec.archive_member


def _disk_usage_bytes(path: Path) -> int:
    """Sum every file size under ``path``; ``0`` if ``path`` does not exist."""
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _format_size(num: int) -> str:
    """Human-readable size with one decimal. Returns ``"-"`` for zero."""
    if num <= 0:
        return "-"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(num)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} {units[-1]}"


def _resolve_one(name: str) -> _ZooAsset:
    for asset in _discover_assets():
        if asset.spec.name == name:
            return asset
    available = ", ".join(a.spec.name for a in _discover_assets()) or "<none>"
    raise typer.BadParameter(f"no asset named {name!r} in the zoo (available: {available})")


@asset_app.command("list", help="List every asset declared under genelab.asset_zoo.")
def list_cmd() -> None:
    from rich.table import Table

    assets = _discover_assets()
    if not assets:
        err_console.print("[warn]no AssetSpec found under genelab.asset_zoo[/]")
        raise typer.Exit(code=1)

    table = Table(title="Asset zoo", show_lines=False, expand=False)
    table.add_column("name", style="bold")
    table.add_column("status")
    table.add_column("size", justify="right")
    table.add_column("module", style="dim")

    for asset in assets:
        _root, entry = _cache_paths(asset.spec)
        downloaded = entry is not None and entry.exists()
        cache_root, _ = _cache_paths(asset.spec)
        size = _disk_usage_bytes(cache_root) if downloaded else 0
        status = "[ok]downloaded[/]" if downloaded else "[muted]not downloaded[/]"
        table.add_row(asset.spec.name, status, _format_size(size), asset.module)

    console.print(table)


@asset_app.command("info", help="Show download spec + cache path for one asset.")
def info_cmd(name: Annotated[str, typer.Argument(help="Asset name (e.g. 'go1').")]) -> None:
    from rich.panel import Panel
    from rich.table import Table

    asset = _resolve_one(name)
    spec = asset.spec
    cache_root, entry = _cache_paths(spec)
    downloaded = entry is not None and entry.exists()

    body = Table.grid(padding=(0, 2))
    body.add_column(style="bold")
    body.add_column()
    body.add_row("module", f"{asset.module}.{asset.attr}")
    body.add_row("url", spec.url)
    body.add_row("md5", spec.md5)
    body.add_row("filename", spec.filename)
    body.add_row(
        "archive_member",
        spec.archive_member if spec.archive_member is not None else "[muted](single file)[/]",
    )
    body.add_row("cache root", str(cache_root))
    if entry is not None:
        body.add_row("entry", str(entry))
    body.add_row(
        "status",
        "[ok]downloaded[/]" if downloaded else "[muted]not downloaded[/]",
    )
    if downloaded:
        body.add_row("size", _format_size(_disk_usage_bytes(cache_root)))

    console.print(Panel(body, title=f"asset: {spec.name}", expand=False))


@asset_app.command(
    "download",
    help="Download one named asset or every asset (with --all). Re-uses the existing md5-verified cache.",
)
def download_cmd(
    name: Annotated[
        str | None,
        typer.Argument(help="Asset name to download (omit when using --all)."),
    ] = None,
    download_all: Annotated[
        bool,
        typer.Option("--all", help="Download every asset declared under genelab.asset_zoo."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-download even if the cached md5 matches."),
    ] = False,
) -> None:
    if download_all and name is not None:
        raise typer.BadParameter("--all is mutually exclusive with a positional NAME")
    if not download_all and name is None:
        raise typer.BadParameter("pass an asset NAME or --all")

    targets = _discover_assets() if download_all else [_resolve_one(name)]  # type: ignore[arg-type]
    if not targets:
        err_console.print("[warn]no assets to download[/]")
        raise typer.Exit(code=1)

    failures: list[tuple[str, str]] = []
    with fetch_progress():
        for asset in targets:
            try:
                path = fetch_asset(asset.spec, force=force)
            except AssetDownloadError as exc:
                failures.append((asset.spec.name, str(exc)))
                continue
            console.print(f"[ok]✓[/] [bold]{asset.spec.name}[/] → {path}")

    if failures:
        for name_, msg in failures:
            err_console.print(f"[error]✗ {name_}[/]: {msg}")
        raise typer.Exit(code=1)


@asset_app.command(
    "purge",
    help="Remove one named asset (or --all) from the local cache.",
)
def purge_cmd(
    name: Annotated[
        str | None,
        typer.Argument(help="Asset name to purge (omit when using --all)."),
    ] = None,
    purge_all: Annotated[
        bool,
        typer.Option("--all", help="Purge every asset declared under genelab.asset_zoo."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt."),
    ] = False,
) -> None:
    if purge_all and name is not None:
        raise typer.BadParameter("--all is mutually exclusive with a positional NAME")
    if not purge_all and name is None:
        raise typer.BadParameter("pass an asset NAME or --all")

    targets = _discover_assets() if purge_all else [_resolve_one(name)]  # type: ignore[arg-type]
    existing = [
        (asset, _cache_paths(asset.spec)[0])
        for asset in targets
        if _cache_paths(asset.spec)[0].exists()
    ]
    if not existing:
        err_console.print("[muted]nothing to purge (no matching cache directory).[/]")
        return

    summary = ", ".join(asset.spec.name for asset, _ in existing)
    if not yes:
        confirmed = typer.confirm(f"Purge cached assets ({summary})?", default=False)
        if not confirmed:
            raise typer.Exit(code=1)

    for asset, root in existing:
        shutil.rmtree(root, ignore_errors=True)
        console.print(f"[ok]✓[/] purged [bold]{asset.spec.name}[/] ({root})")
