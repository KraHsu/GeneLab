"""Deprecated alias for :mod:`genelab.utils.distributed` (ADR-0009 / ROADMAP §9 R7.3b).

The torchrun helpers moved to ``genelab.utils.distributed`` — they are a generic
environment/torchrun utility with no RL-specific content, and a domain module
(``scene``) needs ``pin_cuda_device``, which `rl` may not sit below. Import from
``genelab.utils.distributed`` instead. This shim keeps the old path working for one
release and will be removed in the next minor.
"""

import warnings

from genelab.utils.distributed import (
    global_rank as global_rank,
    is_distributed as is_distributed,
    is_main_process as is_main_process,
    local_rank as local_rank,
    pin_cuda_device as pin_cuda_device,
    shutdown_process_group as shutdown_process_group,
    world_size as world_size,
)

warnings.warn(
    "genelab.rl.distributed is deprecated; import from genelab.utils.distributed.",
    DeprecationWarning,
    stacklevel=2,
)
