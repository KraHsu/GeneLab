"""Deprecated alias for :mod:`genelab.rl.vecenvs.rsl_rl` (ADR-0007 / ROADMAP §9 R6).

Import ``RslRlVecEnvWrapper`` from ``genelab.rl.vecenvs.rsl_rl`` instead. This shim
keeps the old path working for one release and will be removed in the next minor.
"""

import warnings

from genelab.rl.vecenvs.rsl_rl import RslRlVecEnvWrapper as RslRlVecEnvWrapper

warnings.warn(
    "genelab.rl.rsl_rl_wrapper is deprecated; import from genelab.rl.vecenvs.rsl_rl.",
    DeprecationWarning,
    stacklevel=2,
)
