#!/usr/bin/env python3
"""Static calibration check for cube pose vs. hand pose (Genesis-native port).

Ported from wuji-mjlab deploy/reorient/tools/calib_check.py — adapted to the
Genesis reorient scene and GeneLab's deploy helpers.

What this does:
  1. Initialises the hand driver and ramps the hand to the reorient home pose.
  2. Opens the Genesis viewer showing the digital twin (hand + cube).
  3. Loops at ~20 Hz reading joint encoders + cube pose over ZMQ, rendering the
     live hand pose and the observed cube each tick.

Use this to eyeball whether the ArUco-based cube pose estimate (anchored to the
wrist AprilTag world frame) matches the physical cube. There is no policy and no
control beyond the initial homing ramp — the hand stays at home.

Run:
    uv run wuji observer &          # publisher
    uv run wuji calib               # this tool
    uv run wuji calib --mock        # no hardware

Press Ctrl+C or close the viewer window to exit. Needs a GPU + display (Genesis
viewer); the transform math itself is covered by the headless deploy tests.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from genelab_wuji.deploy.hand_driver import HandDriverBase, MockHandDriver
from genelab_wuji.deploy.real2sim import cube_pose_in_tag_to_world
from genelab_wuji.deploy.scripts._env import (
    build_reorient_env,
    set_cube_pose,
    set_hand_joints,
    tag_world_pose,
)
from genelab_wuji.deploy.zmq_bridge import DEFAULT_CUBE_PORT, CubeReceiver


def _loop(env: Any, drv: HandDriverBase, cube_recv: CubeReceiver | None, rate_hz: float) -> None:
    """Render the live hand + observed cube until the viewer closes / Ctrl+C."""
    tag_pos_w, tag_quat_w = tag_world_pose(env)  # fixed-base hand -> constant
    dt = 1.0 / max(rate_hz, 1.0)
    try:
        while not env.viewer_closed:
            set_hand_joints(env, drv.read_encoders())
            if cube_recv is not None:
                cube_pos_tag, cube_quat_tag = cube_recv.latest()
                cube_pos_w, cube_quat_w = cube_pose_in_tag_to_world(
                    tag_pos_w, tag_quat_w, cube_pos_tag, cube_quat_tag
                )
                set_cube_pose(env, cube_pos_w, cube_quat_w)
            # FK-only refresh: render the teleported hand/cube as-set, no physics
            # integration (so the cube isn't pulled down by gravity each frame).
            env.scene.refresh_visualizer()
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[calib-check] interrupted by user.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--cube-port", type=int, default=DEFAULT_CUBE_PORT)
    parser.add_argument("--host", default="localhost")
    parser.add_argument(
        "--no-cube-zmq",
        action="store_true",
        help="skip CubeReceiver — render only the hand (diagnose publisher issues).",
    )
    parser.add_argument("--rate-hz", type=float, default=20.0, help="viewer refresh rate")
    parser.add_argument(
        "--effort-limit", type=float, default=0.5, help="per-joint Nm for the real hand"
    )
    parser.add_argument(
        "--mock", action="store_true", help="use the mock hand (no hardware; renders home pose)"
    )
    args = parser.parse_args(argv)

    cube_recv: CubeReceiver | None = None
    if not args.no_cube_zmq:
        cube_recv = CubeReceiver(port=args.cube_port, host=args.host)
        print(f"[calib-check] CubeReceiver listening on tcp://{args.host}:{args.cube_port}")
    else:
        print("[calib-check] cube ZMQ disabled (--no-cube-zmq)")

    env = build_reorient_env(num_envs=1)  # single viewer; no parallel envs
    try:
        if args.mock:
            drv: HandDriverBase = MockHandDriver()
            drv.home()
            print("[calib-check] mock hand at home — viewer up. Ctrl+C / close window to exit.")
            _loop(env, drv, cube_recv, args.rate_hz)
        else:
            from genelab_wuji.deploy.hand_driver import WujiHandDriver

            with WujiHandDriver(effort_limit_nm=args.effort_limit) as drv:
                print("[calib-check] homing hand ...")
                drv.home(duration_s=3.0)
                print(
                    "[calib-check] homed — viewer up. Compare the rendered cube against the "
                    "real cube. Ctrl+C / close window to exit."
                )
                _loop(env, drv, cube_recv, args.rate_hz)
    finally:
        if cube_recv is not None:
            cube_recv.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
