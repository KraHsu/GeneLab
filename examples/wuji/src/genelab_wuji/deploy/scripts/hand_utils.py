#!/usr/bin/env python3
"""Wuji-hand hardware utilities — the first sim2real step (Genesis-native port).

Single entry-point for low-level hand operations, mirroring wuji-mjlab's
``deploy/reorient/scripts/hand_utils.py``. Two subcommands:

  check  READ-ONLY connection + encoder sanity check. Does not write any
         targets; the hand stays where it is. Run this FIRST.
  home   Ramp all 20 joints to the reorient home grasp pose via a 3s ease-in-out
         (``WujiHandDriver.home``). Enables joints on entry, disables on exit,
         and reports tracking error.

Usage:
    python -m genelab_wuji.deploy.scripts.hand_utils check
    python -m genelab_wuji.deploy.scripts.hand_utils home

The home pose is ``REORIENT_JOINT_POS`` (via ``config.default_joint_pos``) — the
same grasp keyframe the policy and ``MockHandDriver`` start from, so after ``home``
the real hand matches the sim reset state. Needs the optional ``wujihandpy`` hand
SDK; it is imported lazily so this module stays importable without hardware.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from genelab_wuji.deploy.config import JOINT_NAMES_20, N_JOINTS, default_joint_pos


def _print_finger_major(label: str, qpos_rad: np.ndarray) -> None:
    """Print a flat (20,) qpos as 5 fingers x 4 joints, in degrees."""
    print(label)
    for i in range(5):
        finger_deg = np.rad2deg(qpos_rad[4 * i : 4 * (i + 1)])
        print(f"  finger{i + 1}: " + ", ".join(f"{v:+6.1f}°" for v in finger_deg))


def cmd_home(_args: argparse.Namespace) -> int:
    """Ramp the real hand to the home grasp pose over a 3s ease-in-out.

    Safety:
      - ``WujiHandDriver`` context manager: enable joints on enter, disable on exit.
      - ``home()`` does a 50 Hz smoothstep ramp from the current pose (no snap).
      - Reads back the actual position and reports max / RMS error vs target.

    After this the hand is at ``default_joint_pos()`` — the pose the sim resets to.
    """
    from genelab_wuji.deploy.hand_driver import WujiHandDriver

    home_qpos = default_joint_pos()
    print("=" * 60)
    print("Wuji hand HOME — 3s smooth ramp to the reorient grasp pose")
    print("=" * 60)
    _print_finger_major("\nTarget pose (default_joint_pos, degrees, finger-major):", home_qpos)

    with WujiHandDriver() as drv:
        print("\nReading current pose...")
        current = drv.read_encoders()
        max_diff = float(np.abs(current - home_qpos).max())
        print(f"  Max diff from target: {max_diff:.3f} rad ({np.rad2deg(max_diff):.1f}°)")

        print("\nRamping to home pose over 3s...")
        drv.home(duration_s=3.0)

        print("\nReading actual after ramp...")
        actual = drv.read_encoders()
        err = np.abs(actual - home_qpos)
        max_err = float(err.max())
        rms_err = float(np.sqrt(np.mean((actual - home_qpos) ** 2)))
        print(f"  Max err: {max_err:.3f} rad ({np.rad2deg(max_err):.2f}°)")
        print(f"  RMS err: {rms_err:.3f} rad ({np.rad2deg(rms_err):.2f}°)")
        if max_err < np.deg2rad(2):
            print("  ✓ Within 2° — home reached")
        elif max_err < np.deg2rad(5):
            print("  ⚠ Within 5° — hand may not be tracking perfectly")
        else:
            print("  ✗ Over 5° error — investigate")

    print("\n✓ hand_utils home complete; joints disabled.")
    return 0


def cmd_check(_args: argparse.Namespace) -> int:
    """READ-ONLY connection + encoder sanity check.

    Connects to ``wujihandpy.Hand()``, reads joint limits + encoders, prints the
    current pose against the expected joint order, and compares to the home pose.
    Does NOT write any targets; the hand stays where it is. This is the safe first
    step before any closed-loop deployment.
    """
    try:
        import wujihandpy
    except ImportError:
        print("ERROR: wujihandpy not installed (the optional hardware SDK).")
        return 1

    print("=" * 60)
    print("Wuji hand connection check (READ-ONLY)")
    print("=" * 60)

    print("\n[1/4] Connecting to wujihandpy.Hand()...")
    try:
        hand = wujihandpy.Hand()
    except Exception as e:  # noqa: BLE001 — surface any hardware error to the operator
        print(f"  FAIL: {e}")
        print("  Check: USB connected? udev rules? hand powered?")
        return 1
    print("  ✓ connected")

    print("\n[2/4] Reading joint limits (no enable / no write)...")
    try:
        upper = hand.read_joint_upper_limit()  # (5, 4)
        lower = hand.read_joint_lower_limit()
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL reading limits: {e}")
        return 1
    print(f"  shape: upper={upper.shape}, lower={lower.shape}")
    print(f"  upper range: [{upper.min():.2f}, {upper.max():.2f}] rad")
    print(f"  lower range: [{lower.min():.2f}, {lower.max():.2f}] rad")

    print("\n[3/4] Reading encoder actual position...")
    try:
        actual = hand.read_joint_actual_position()  # (5, 4)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL: {e}")
        return 1
    flat = np.asarray(actual, dtype=float).flatten()
    print(f"  shape: {actual.shape} → flat ({flat.shape[0]},)")
    print(f"  range: [{flat.min():.3f}, {flat.max():.3f}] rad")
    _print_finger_major("  values (degrees, finger-major):", flat)

    print("\n[4/4] Validating joint order + home offset...")
    print(f"  Expected joint order ({N_JOINTS} joints):")
    for i, name in enumerate(JOINT_NAMES_20):
        print(f"    [{i:2d}] {name}: {np.rad2deg(flat[i]):+6.1f}°")

    expected_home = default_joint_pos()
    diff = flat - expected_home
    print("\n  Current vs home pose (default_joint_pos):")
    print(f"    Max abs diff: {np.abs(diff).max():.3f} rad ({np.rad2deg(np.abs(diff).max()):.1f}°)")
    print(f"    RMS diff:     {np.sqrt(np.mean(diff**2)):.3f} rad")
    print("    (A large diff just means the hand isn't at home yet — not an error.)")

    print("\n" + "=" * 60)
    print("✓ All read operations succeeded. Hand bridge healthy.")
    print("✓ Next: ramp to home with `python -m genelab_wuji.deploy.scripts.hand_utils home`")
    print("=" * 60)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wuji hand hardware utilities (home / check).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_home = sub.add_parser(
        "home", help="Ramp all 20 joints to the home grasp pose (writes targets)."
    )
    p_home.set_defaults(func=cmd_home)
    p_check = sub.add_parser("check", help="READ-ONLY connection + encoder sanity check.")
    p_check.set_defaults(func=cmd_check)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
