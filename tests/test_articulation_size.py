"""Size guard for the ``entity`` articulation modules.

The articulation implementation is split across an ``Articulation``
facade and three private collaborators behind it — the read seam
(``_articulation_state``), write seam (``_articulation_writer``), and binding seam
(``_articulation_binder``). This guard keeps any one of them from quietly growing
back toward a god-class; the thresholds were retargeted down (from 700 / 1000) once
the decomposition shrank the facade.

For each module:

* ``<= WARN_LOC`` (450): silent pass.
* ``WARN_LOC < loc <= FAIL_LOC``: pass, but emit a ``UserWarning`` so the growth
  appears in pytest's warnings summary and can be reviewed.
* ``> FAIL_LOC`` (700): hard failure — review its structure before further growth.

Thresholds are line counts, equivalent to ``wc -l``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

_ENTITY_DIR = Path(__file__).resolve().parents[1] / "src" / "genelab" / "entity"

ARTICULATION_MODULES = (
    "articulation.py",
    "_articulation_state.py",
    "_articulation_writer.py",
    "_articulation_binder.py",
)

WARN_LOC = 450
FAIL_LOC = 700


@pytest.mark.parametrize("filename", ARTICULATION_MODULES)
def test_articulation_module_size_within_budget(filename: str) -> None:
    path = _ENTITY_DIR / filename
    assert path.is_file(), f"expected {path} to exist"
    loc = len(path.read_text(encoding="utf-8").splitlines())

    assert loc <= FAIL_LOC, (
        f"entity/{filename} is {loc} LoC (> {FAIL_LOC}); "
        "review its structure before allowing further growth."
    )

    if loc > WARN_LOC:
        warnings.warn(
            f"entity/{filename} is {loc} LoC (> {WARN_LOC}); "
            "review whether the module should be split or otherwise simplified.",
            UserWarning,
            stacklevel=2,
        )
