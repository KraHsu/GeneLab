"""Project-local path constants.

The cache directory lives at the repository root (``<repo>/.cache``). These constants
live in ``utils`` (the bottom infrastructure band) so ``utils.download`` can resolve the
asset cache without importing up into ``genelab.cache`` — ``genelab.cache`` re-exports
them and owns the cache-population side effects.
"""

from pathlib import Path

# src/genelab/utils/paths.py -> parents: [0]=utils, [1]=genelab, [2]=src, [3]=repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = PROJECT_ROOT / ".cache"
