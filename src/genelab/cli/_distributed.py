"""Distributed-training plumbing for ``genelab train --gpus N``.

Self-contained argv surgery + torchrun relaunch, factored out of
``cli/__init__.py``. The CLI dispatcher
(``_dispatch_train``) and the multi-seed orchestrator are the only callers;
nothing here imports ``genelab.cli`` itself, so no import cycle is introduced.

The helpers split into two groups:

* **argv surgery** — ``_strip_flag_value_pairs`` / ``_strip_distributed_flags`` /
  ``_extract_log_dir_flag`` / ``_has_log_dir_flag`` rewrite a forwarded argv slice
  so each torchrun worker sees an authoritative, deduplicated invocation.
* **env-count resolution + relaunch** — ``_resolve_per_rank_num_envs`` converts
  ``--num-envs`` / ``--num-envs-per-gpu`` into a per-rank count, and
  ``_relaunch_under_torchrun`` re-execs the current ``train`` under
  ``torch.distributed.run``.
"""

import os
import sys
from pathlib import Path
from typing import Any, Final

# These keep their leading underscores (they are CLI-package-private, re-exported
# through ``cli/__init__.py``) but are this module's external API — listing them in
# ``__all__`` marks them exported so they are not flagged as unused at the def site.
__all__ = [
    "_extract_log_dir_flag",
    "_has_log_dir_flag",
    "_relaunch_under_torchrun",
    "_resolve_per_rank_num_envs",
    "_strip_distributed_flags",
    "_strip_flag_value_pairs",
]

_STRIPPABLE_DISTRIBUTED_FLAGS: Final[frozenset[str]] = frozenset(
    {"--gpus", "--num-envs", "--num_envs", "--num-envs-per-gpu", "--num_envs_per_gpu"}
)


def _strip_flag_value_pairs(tokens: list[str], flags: frozenset[str]) -> list[str]:
    """Drop ``--flag VALUE`` and ``--flag=VALUE`` entries for every flag in ``flags``."""
    out: list[str] = []
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in flags:
            skip_next = True
            continue
        if any(tok.startswith(f"{flag}=") for flag in flags):
            continue
        out.append(tok)
    return out


def _strip_distributed_flags(tokens: list[str]) -> list[str]:
    """Drop ``--gpus`` / ``--num-envs`` / ``--num-envs-per-gpu`` (and the ``_``-spelt forms),
    in both ``--flag value`` and ``--flag=value`` shapes, from a forwarded argv slice."""
    return _strip_flag_value_pairs(tokens, _STRIPPABLE_DISTRIBUTED_FLAGS)


def _extract_log_dir_flag(tokens: list[str]) -> Path | None:
    """Return the ``--log-dir`` / ``--log_dir`` value from a token slice, if present."""
    for i, tok in enumerate(tokens):
        if tok in {"--log-dir", "--log_dir"} and i + 1 < len(tokens):
            return Path(tokens[i + 1])
        if tok.startswith("--log-dir=") or tok.startswith("--log_dir="):
            return Path(tok.split("=", 1)[1])
    return None


def _has_log_dir_flag(tokens: list[str]) -> bool:
    for t in tokens:
        if t in {"--log-dir", "--log_dir"}:
            return True
        if t.startswith("--log-dir=") or t.startswith("--log_dir="):
            return True
    return False


def _resolve_per_rank_num_envs(runner_args: dict[str, str], *, gpus: int) -> int | None:
    """Pop ``num_envs`` / ``num_envs_per_gpu`` from ``runner_args`` and return per-rank N.

    ``--num-envs N`` is interpreted as the **total** across all ranks: ``N // gpus``
    becomes the per-rank count and ``N`` must be divisible by ``gpus``.
    ``--num-envs-per-gpu M`` is verbatim per-rank. Passing both is a hard error so users
    don't quietly get one or the other's semantics. Returns ``None`` when neither flag
    was set so callers can defer to the cfg default.
    """
    total_raw = runner_args.pop("num_envs", None)
    per_gpu_raw = runner_args.pop("num_envs_per_gpu", None)
    if total_raw is not None and per_gpu_raw is not None:
        raise SystemExit(
            "--num-envs and --num-envs-per-gpu are mutually exclusive; pass exactly "
            "one (or neither, to use the task's cfg default)."
        )
    if per_gpu_raw is not None:
        return int(per_gpu_raw)
    if total_raw is None:
        return None
    total = int(total_raw)
    if gpus <= 0:
        raise SystemExit(f"--gpus must be a positive integer (got {gpus})")
    if total % gpus != 0:
        raise SystemExit(
            f"--num-envs {total} is not divisible by --gpus {gpus}; pick a multiple "
            f"of {gpus} or use --num-envs-per-gpu instead."
        )
    return total // gpus


def _relaunch_under_torchrun(
    gpus: int,
    agent_cfg: Any,
    runner_args: dict[str, str],
    num_envs_per_rank: int | None,
    *,
    task_id: str,
) -> None:
    """Re-exec the current ``train`` invocation under torchrun.

    The parent precomputes the log directory and forwards it via ``--log-dir`` so all
    ranks land in the same directory (avoiding per-rank timestamp drift). The original
    argv is forwarded verbatim minus the ``--gpus`` / ``--num-envs`` / ``--num-envs-per-gpu``
    tokens; if either env-count flag was set, an authoritative ``--num-envs-per-gpu N``
    is injected so each worker sees the parent-resolved per-rank value. The resolved
    ``task_id`` is also injected when missing from the forwarded argv — otherwise a
    parent-side interactive pick would force every torchrun worker back into the
    questionary picker, producing repeated CPR-probe warnings (and a blocked launch).
    """
    from genelab.rl.runner import resolve_log_dir

    log_root_raw = runner_args.get("log_dir")
    log_root = Path(log_root_raw) if log_root_raw else Path("logs") / "rsl_rl"
    log_dir = resolve_log_dir(log_root, agent_cfg.experiment_name, agent_cfg.run_name)

    inner = _strip_distributed_flags(sys.argv[1:])
    if task_id not in inner:
        try:
            insert_at = inner.index("train") + 1
        except ValueError:
            insert_at = 0
        inner.insert(insert_at, task_id)
    if num_envs_per_rank is not None:
        inner += ["--num-envs-per-gpu", str(num_envs_per_rank)]
    if not _has_log_dir_flag(inner):
        inner += ["--log-dir", str(log_dir)]

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpus}",
        "-m",
        "genelab.cli",
        *inner,
    ]
    os.execvp(cmd[0], cmd)
