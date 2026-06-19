# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
# Ported into GeneLab from wuji-mjlab deploy/reorient/lib/cube_geom.py
"""Shared helpers for cube geometry: cube_tags JSON resolution + runtime scaling.

Used by both the vision pipeline (cube_world_observer.py — needs cube/tag sizes
for AprilTag/ArUco PnP) and the sim/benchmark pipeline (play_real.py — needs to
patch the MuJoCo cube body so visualization and physics match the real cube).
"""

from __future__ import annotations

import json
import os
from typing import Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CUBE_CONFIG_FILE = os.path.join(ROOT_DIR, "config", "cube_tags.json")


def resolve_cube_config_path(arg: str | None) -> str:
    """Resolve a --cube CLI argument to an absolute cube_tags JSON path.

    Accepts:
      - None / "" / "default"  -> ``config/cube_tags.json`` (the 54mm baseline).
      - An existing path (absolute or relative to cwd) -> used as-is.
      - A short size suffix like ``"36"`` / ``"40_5"`` -> ``config/cube_tags<suffix>.json``.

    Raises FileNotFoundError if the resolved path does not exist.
    """
    if not arg or arg == "default":
        return DEFAULT_CUBE_CONFIG_FILE
    if os.path.exists(arg):
        return os.path.abspath(arg)
    candidate = os.path.join(ROOT_DIR, "config", f"cube_tags{arg}.json")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"--cube={arg!r}: not a path nor a known size suffix. Tried {candidate!r}."
    )


def load_cube_config(path: str) -> dict[str, Any]:
    """Load a cube_tags JSON file."""
    with open(path) as f:
        return json.load(f)
