"""Pin each torchrun worker to a single GPU before any CUDA init.

Must be imported BEFORE torch / genesis. Quadrants (Genesis's compute backend)
binds to whichever GPUs are visible at process start and ignores
``torch.cuda.set_device()``. Without this hook, every rank allocates its
Quadrants tensors on ``cuda:0`` while ``gs.device`` is ``cuda:{LOCAL_RANK}``,
producing a device-mismatch error during scene build.
"""

import os


def pin_visible_device_for_rank() -> None:
    """Restrict the current worker to a single physical GPU via ``CUDA_VISIBLE_DEVICES``.

    No-op outside torchrun (``WORLD_SIZE <= 1``). When applied:
      * picks the ``LOCAL_RANK``-th entry of any existing ``CUDA_VISIBLE_DEVICES``
        (so users restricting GPUs upstream still partition correctly), else
        sets ``CUDA_VISIBLE_DEVICES = LOCAL_RANK``;
      * rewrites ``LOCAL_RANK=0`` because the rank now sees its assigned GPU as
        ``cuda:0`` and rsl_rl's ``OnPolicyRunner`` requires
        ``device == f"cuda:{LOCAL_RANK}"``;
      * leaves ``RANK`` and ``WORLD_SIZE`` untouched so NCCL rendezvous and
        ``genelab.utils.distributed.is_main_process`` keep working.

    Must run exactly once per worker, before any module imports ``torch``. The
    sole call site is the top of ``genelab/cli/__main__.py``.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return
    lr_raw = os.environ.get("LOCAL_RANK")
    if lr_raw is None:
        return
    lr = int(lr_raw)
    existing = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if existing:
        devices = [d.strip() for d in existing.split(",") if d.strip()]
        if lr >= len(devices):
            raise RuntimeError(
                f"LOCAL_RANK={lr} exceeds visible device count ({len(devices)}); "
                f"CUDA_VISIBLE_DEVICES={existing!r}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[lr]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(lr)
    os.environ["LOCAL_RANK"] = "0"
