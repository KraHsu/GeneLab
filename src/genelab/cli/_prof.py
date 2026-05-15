"""``genelab prof`` — utilities for working with ``torch.profiler`` traces."""

import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer

prof_app = typer.Typer(
    name="prof",
    help="Inspect and open torch.profiler traces.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
)


_DEFAULT_LOG_DIR = Path("logs/torch_profile")


@prof_app.command("open", help="Launch TensorBoard against a profiler log directory.")
def open_cmd(
    log_dir: Annotated[
        Path,
        typer.Argument(
            metavar="LOG_DIR",
            help="Directory holding torch.profiler traces. Defaults to logs/torch_profile.",
        ),
    ] = _DEFAULT_LOG_DIR,
    port: Annotated[
        int,
        typer.Option("--port", help="TCP port TensorBoard binds to."),
    ] = 6006,
    host: Annotated[
        str,
        typer.Option("--host", help="Host interface TensorBoard binds to."),
    ] = "127.0.0.1",
) -> None:
    if not log_dir.exists():
        raise SystemExit(f"profile log directory not found: {log_dir}")
    if shutil.which("tensorboard") is None:
        raise SystemExit("tensorboard is not on PATH; install with `uv pip install tensorboard`")
    subprocess.run(
        ["tensorboard", "--logdir", str(log_dir), "--port", str(port), "--host", host],
        check=False,
    )
