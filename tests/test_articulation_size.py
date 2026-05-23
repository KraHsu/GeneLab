"""Size guard for ``entity/articulation.py``.

This test keeps the articulation implementation from growing unnoticed.

It does not force an immediate split. Instead, it makes file growth visible:

* ``<= WARN_LOC`` (700): silent pass.
* ``WARN_LOC < loc <= FAIL_LOC``: pass, but emit a ``UserWarning`` so the growth
  appears in pytest's warnings summary and can be reviewed.
* ``> FAIL_LOC`` (1000): hard failure. At that point the file is large enough
  that its structure should be reviewed before allowing further growth.

Thresholds are line counts, equivalent to ``wc -l``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

ARTICULATION = (
    Path(__file__).resolve().parents[1] / "src" / "genelab" / "entity" / "articulation.py"
)

WARN_LOC = 700
FAIL_LOC = 1000


def test_articulation_file_size_within_budget() -> None:
    assert ARTICULATION.is_file(), f"expected {ARTICULATION} to exist"
    loc = len(ARTICULATION.read_text(encoding="utf-8").splitlines())

    assert loc <= FAIL_LOC, (
        f"entity/articulation.py is {loc} LoC (> {FAIL_LOC}); "
        "review its structure before allowing further growth."
    )

    if loc > WARN_LOC:
        warnings.warn(
            f"entity/articulation.py is {loc} LoC (> {WARN_LOC}); "
            "review whether the file should be split or otherwise simplified.",
            UserWarning,
            stacklevel=2,
        )
