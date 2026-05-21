"""Shared regex-based joint-name → indices matcher for action terms.

Both :class:`BinaryGripperAction` and :class:`ContinuousGripperAction` need
the same code to translate a list of joint-name regex patterns into a
deduplicated list of joint indices. The two ``__init__`` bodies were
jaccard 1.000 copies of each other; per ADR-0003 / R2.3 the matching
logic lives here and each gripper raises its own zero-match error with
the term name preserved.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

__all__ = ["match_joints"]


def match_joints(patterns: Sequence[str], joint_names: Sequence[str]) -> list[int]:
    """Return deduplicated joint indices for joints matching any pattern.

    Each ``pat`` in ``patterns`` is compiled as a regex; on ``re.error``
    it falls back to literal matching via ``re.escape(pat)``. A joint
    matches when the (possibly-escaped) regex either ``fullmatch``-es
    or ``search``-es the joint name. Indices are preserved in the order
    they are first encountered; duplicates across patterns are skipped.
    """
    matched: list[int] = []
    for pat in patterns:
        try:
            regex = re.compile(pat)
        except re.error:
            regex = re.compile(re.escape(pat))
        for i, name in enumerate(joint_names):
            if (regex.fullmatch(name) or regex.search(name)) and i not in matched:
                matched.append(i)
    return matched
