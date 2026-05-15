"""Rich-based progress reporter for :func:`genelab.utils.download.fetch_asset`.

The download helper itself stays UI-agnostic and only invokes a
:class:`ProgressCallback`. This module is the CLI's adapter: it installs a
Rich-backed callback into :data:`genelab.utils.download.PROGRESS_CALLBACK` for the
duration of a ``with fetch_progress():`` block so any asset fetched while building
task / env / robot configs renders a per-asset download bar on stderr.

Single concurrent bar suffices today — asset-zoo factories are called serially during
cfg construction, never in threads. If that ever changes the Rich Progress widget will
DTRT (it tracks tasks by id).
"""

from collections.abc import Generator
from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from genelab.utils.download import PROGRESS_CALLBACK


@contextmanager
def fetch_progress(console: Console | None = None) -> Generator[None, None, None]:
    """Install a Rich-rendered progress reporter for the duration of the block.

    Nested usage is a no-op — the inner block sees the outer's ContextVar binding and
    skips reinstalling. Pass ``console`` to render onto a specific :class:`Console`;
    defaults to a stderr console so stdout output (e.g. ``info`` table) stays clean
    for downstream piping.
    """

    if PROGRESS_CALLBACK.get() is not None:
        yield
        return

    target = console if console is not None else Console(stderr=True)
    progress = Progress(
        TextColumn("[bold blue]fetch[/] {task.fields[asset]}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=target,
        transient=True,
    )
    task_ids: dict[str, TaskID] = {}

    def _callback(done: int, total: int | None, *, name: str) -> None:
        task_id = task_ids.get(name)
        if task_id is None:
            task_id = progress.add_task("download", asset=name, total=total)
            task_ids[name] = task_id
        progress.update(task_id, completed=done, total=total)

    token = PROGRESS_CALLBACK.set(_callback)
    progress.start()
    try:
        yield
    finally:
        progress.stop()
        PROGRESS_CALLBACK.reset(token)
