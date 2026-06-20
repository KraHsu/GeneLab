"""Unified ``wuji`` command-line entry for the deploy + real2sim toolchain.

Installed as a console script by the ``examples/wuji[deploy]`` extra, so the
whole pipeline runs through one verb instead of long ``python -m`` module paths::

    uv run wuji <command> [options]
    uv run wuji play --ckpt policy.onnx --real --goal-mode random
    uv run wuji <command> --help        # per-command options

Each command lazily imports its target module so that, e.g., ``wuji play`` does
not pull in the Hikvision MVS SDK that ``wuji observer`` needs (and vice-versa) —
the observer module raises at import time when the SDK is absent.
"""

from __future__ import annotations

import importlib
import sys

# command -> (module, prefix-args prepended before user args)
# prefix-args let two verbs share one module (hand_utils' check / home subparser).
_COMMANDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "check": ("genelab_wuji.deploy.scripts.hand_utils", ("check",)),
    "home": ("genelab_wuji.deploy.scripts.hand_utils", ("home",)),
    "observer": ("genelab_wuji.deploy.scripts.cube_world_observer", ()),
    "viewer": ("genelab_wuji.deploy.scripts.toreal_viewer", ()),
    "calib": ("genelab_wuji.deploy.scripts.calib_check", ()),
    "play": ("genelab_wuji.deploy.scripts.play_real", ()),
}

_HELP: dict[str, str] = {
    "check": "read-only hand-bridge test (connection + encoder sanity)",
    "home": "ease-in-out ramp the hand to the grasp pose",
    "observer": "Hikvision camera -> ArUco cube pose, published on ZMQ",
    "viewer": "real2sim Genesis mirror of the observed cube",
    "calib": "calibration viewer: live hand + observed cube vs. digital twin",
    "play": "deploy control loop (real/mock) + goal modes + success monitor",
}


def _usage() -> str:
    width = max(len(c) for c in _COMMANDS)
    lines = ["usage: wuji <command> [options]", "", "commands:"]
    lines += [f"  {name:<{width}}  {_HELP[name]}" for name in _COMMANDS]
    lines += ["", "run `wuji <command> --help` for per-command options."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(_usage(), file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if cmd not in _COMMANDS:
        print(f"wuji: unknown command {cmd!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    module_name, prefix = _COMMANDS[cmd]
    module = importlib.import_module(module_name)
    return int(module.main([*prefix, *rest]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
