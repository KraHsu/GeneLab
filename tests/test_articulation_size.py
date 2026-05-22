"""Size guard for ``entity/articulation.py`` (ADR-0010 §Risks R10.1 / Validation).

ADR-0010 deliberately *defers* the entity/articulation god-class split: the
seams aren't validated without a second articulation-like entity type, so
splitting now risks guessing the boundaries wrong. The accepted cost is that
the file stays large in the medium term. The recorded risk (R10.1) is that it
keeps accreting past the point where a split would be cheap, *unnoticed*.

This test is the soft check that risk calls for. It does **not** force the
split — it makes the file's growth a conversation:

* ``<= WARN_LOC`` (700): silent pass.
* ``WARN_LOC < loc <= FAIL_LOC``: pass, but emit a ``UserWarning`` so the
  growth shows in pytest's warnings summary — a prompt to revisit ADR-0010's
  trigger criteria (a second entity type, a partial-``RobotState`` consumer, a
  test-isolation pain) and decide whether the seam is now visible.
* ``> FAIL_LOC`` (1000): hard failure. By then the file is unambiguously a
  god-class and the "defer" decision must be re-litigated via an RFC + a
  superseding ADR (per ADR-0010 §Rollback).

Thresholds are line counts (``wc -l`` equivalent), matching how the ADR sized
the file (528 LoC at the time of writing).
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
        f"entity/articulation.py is {loc} LoC (> {FAIL_LOC}); the ADR-0010 "
        "'defer the split' decision must now be re-litigated — open an RFC + a "
        "superseding ADR (see ADR-0010 §Rollback) rather than letting it grow."
    )

    if loc > WARN_LOC:
        warnings.warn(
            f"entity/articulation.py is {loc} LoC (> {WARN_LOC}); revisit "
            "ADR-0010's split trigger criteria — the file is accreting and a "
            "seam may now be visible.",
            UserWarning,
            stacklevel=2,
        )
