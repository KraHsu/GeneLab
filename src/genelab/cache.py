"""Project-local cache directory helpers."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache"
XDG_CACHE_DIR = CACHE_DIR / "xdg"
MATPLOTLIB_CACHE_DIR = CACHE_DIR / "matplotlib"


def ensure_project_cache() -> None:
    """Create project cache folders and route common simulation caches there."""

    XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (XDG_CACHE_DIR / "genesis").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
    os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
