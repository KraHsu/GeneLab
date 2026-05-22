"""Deprecated alias for :mod:`genelab.rl.vecenvs.sb3` (ADR-0007 / ROADMAP §9 R6).

Import ``GenelabSb3VecEnv`` from ``genelab.rl.vecenvs.sb3`` instead. This shim keeps
the old path working for one release and will be removed in the next minor.
"""

import warnings

from genelab.rl.vecenvs.sb3 import GenelabSb3VecEnv as GenelabSb3VecEnv

warnings.warn(
    "genelab.rl.sb3_wrapper is deprecated; import from genelab.rl.vecenvs.sb3.",
    DeprecationWarning,
    stacklevel=2,
)
