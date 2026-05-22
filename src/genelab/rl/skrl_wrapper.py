"""Deprecated alias for :mod:`genelab.rl.vecenvs.skrl` (ADR-0007 / ROADMAP §9 R6).

Import ``GenelabSkrlWrapper`` from ``genelab.rl.vecenvs.skrl`` instead. This shim
keeps the old path working for one release and will be removed in the next minor.
"""

import warnings

from genelab.rl.vecenvs.skrl import GenelabSkrlWrapper as GenelabSkrlWrapper

warnings.warn(
    "genelab.rl.skrl_wrapper is deprecated; import from genelab.rl.vecenvs.skrl.",
    DeprecationWarning,
    stacklevel=2,
)
