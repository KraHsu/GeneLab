"""EvalCallback — periodic in-training evaluation + ``best_model`` selection.

Implemented as a backend-agnostic outer loop that calls ``backend.train()`` in
chunks of ``eval_every_iters`` iterations, then loads the most recent checkpoint
and runs a deterministic eval through :func:`genelab.rl.evaluator.run_evaluation`.
When the eval return-mean improves on the prior best, the checkpoint is copied to
``<log_dir>/best_model.<ext>`` and ``<log_dir>/best_model_meta.json`` is updated.

Caveats (and why the chunk-driven design is acceptable):

* Each chunk closes and rebuilds the Genesis env via the backend's normal train
  lifecycle. Genesis init costs amortize when ``eval_every_iters`` >> "iters per
  Genesis init". Set ``eval_every_iters`` to ≥ 50 for short tasks.
* For off-policy algorithms (SAC / TD3 / DDPG via skrl or sb3), reloading from a
  checkpoint between chunks loses the replay buffer. Tasks tolerate this but
  sample efficiency degrades. See ROADMAP M2 for a possible callback-API fix.
"""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class EvalCallbackCfg:
    """Settings for periodic in-training eval.

    ``enabled=False`` keeps the legacy single-shot ``backend.train()`` path; set
    via ``--eval-every K`` on the CLI to opt into chunked training.
    """

    enabled: bool = False
    eval_every_iters: int = 100
    eval_episodes: int = 10
    eval_num_envs: int | None = None  # None = same as train num_envs
    eval_seed: int = 0


def _checkpoint_extension(backend_name: str) -> str:
    """Return the on-disk extension for ``backend_name``'s checkpoints."""
    return ".zip" if backend_name == "sb3" else ".pt"


def _find_latest_checkpoint(log_dir: Path, backend_name: str) -> Path | None:
    """Return the newest checkpoint file written to ``log_dir`` by ``backend_name``.

    Each backend uses a different layout:

    * **rsl_rl**: ``model_<iter>.pt`` written directly under ``log_dir``.
    * **skrl**: ``checkpoints/agent_<iter>.pt`` under ``log_dir``.
    * **sb3**: ``model_<steps>_steps.zip`` (CheckpointCallback) or ``model.zip``
      (final save) under ``log_dir``.

    Returns ``None`` when no checkpoint matching the backend's pattern is found.
    """
    ext = _checkpoint_extension(backend_name)
    candidates: list[Path] = []
    if backend_name == "skrl":
        ck_dir = log_dir / "checkpoints"
        if ck_dir.is_dir():
            candidates.extend(ck_dir.glob(f"*{ext}"))
    else:
        candidates.extend(log_dir.glob(f"*{ext}"))
    candidates = [p for p in candidates if not p.name.startswith("best_model")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _promote_best(
    ckpt: Path, log_dir: Path, backend_name: str, eval_payload: dict[str, Any]
) -> Path:
    """Copy ``ckpt`` to ``<log_dir>/best_model.<ext>`` and dump matching meta JSON."""
    ext = _checkpoint_extension(backend_name)
    best_path = log_dir / f"best_model{ext}"
    shutil.copy2(ckpt, best_path)
    meta = {
        "source_checkpoint": str(ckpt),
        **eval_payload,
    }
    (log_dir / "best_model_meta.json").write_text(json.dumps(meta, indent=2))
    return best_path


def run_with_eval_callback(
    *,
    task_id: str,
    train_chunk: Any,
    eval_cfg: EvalCallbackCfg,
    total_iterations: int,
    log_dir: Path,
    backend_name: str,
    train_num_envs: int,
) -> Path:
    """Drive ``train_chunk(iterations, resume_from)`` in eval-every chunks.

    Parameters

    * ``train_chunk`` — callable that runs the backend for ``iterations`` iterations
      starting from ``resume_from`` (Path | None) and returns the resolved log
      directory. The runner.train_task wrapper supplies this so this module is
      backend-agnostic.
    * ``eval_cfg`` — periodic-eval settings; ``eval_cfg.enabled`` must be ``True``.
    * ``total_iterations`` — overall iteration budget, chunked.
    * ``backend_name`` — used to pick the checkpoint glob pattern (.pt vs .zip).

    Returns the final ``log_dir``.
    """
    from genelab.cli._eval import eval_task

    assert eval_cfg.enabled, "run_with_eval_callback requires enabled cfg"
    log_dir.mkdir(parents=True, exist_ok=True)

    best_return = float("-inf")
    best_payload: dict[str, Any] | None = None
    last_ckpt: Path | None = None
    completed = 0
    eval_num_envs = eval_cfg.eval_num_envs or train_num_envs

    while completed < total_iterations:
        chunk = min(eval_cfg.eval_every_iters, total_iterations - completed)
        _logger.info(
            "eval-callback: training chunk %d/%d (resume=%s)",
            completed + chunk,
            total_iterations,
            last_ckpt,
        )
        train_chunk(chunk, last_ckpt)
        completed += chunk

        ckpt = _find_latest_checkpoint(log_dir, backend_name)
        if ckpt is None:
            _logger.warning(
                "eval-callback: no checkpoint found in %s after chunk; skipping eval", log_dir
            )
            continue
        last_ckpt = ckpt
        try:
            _, payload = eval_task(
                task_id,
                ckpt,
                num_envs=eval_num_envs,
                episodes=eval_cfg.eval_episodes,
                seed=eval_cfg.eval_seed,
                deterministic=True,
                out_path=None,
            )
        except Exception:
            _logger.exception("eval-callback: eval failed for %s; continuing training", ckpt)
            continue
        return_mean = payload["metrics"]["return_mean"]
        _logger.info(
            "eval-callback: iter=%d ckpt=%s return_mean=%.3f (best=%.3f)",
            completed,
            ckpt.name,
            return_mean,
            best_return if best_return > float("-inf") else float("nan"),
        )
        if return_mean > best_return:
            best_return = return_mean
            best_payload = {"iter": completed, **payload}
            _promote_best(ckpt, log_dir, backend_name, best_payload)

    if best_payload is not None:
        _logger.info(
            "eval-callback: best return_mean=%.3f at iter=%d (saved best_model%s)",
            best_return,
            best_payload["iter"],
            _checkpoint_extension(backend_name),
        )
    return log_dir
