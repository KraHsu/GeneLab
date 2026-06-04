"""Project-local cache directory helpers.

``PROJECT_ROOT`` / ``CACHE_DIR`` live in :mod:`genelab.utils.paths` (the bottom infra
band) so ``utils.download`` can resolve the asset cache without importing up into this
module; they are re-exported here for back-compat.
"""

import os

from genelab.utils.paths import CACHE_DIR as CACHE_DIR, PROJECT_ROOT as PROJECT_ROOT

XDG_CACHE_DIR = CACHE_DIR / "xdg"
MATPLOTLIB_CACHE_DIR = CACHE_DIR / "matplotlib"


def ensure_project_cache() -> None:
    """Create project cache folders and route common simulation caches there."""

    XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (XDG_CACHE_DIR / "genesis").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
    os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
