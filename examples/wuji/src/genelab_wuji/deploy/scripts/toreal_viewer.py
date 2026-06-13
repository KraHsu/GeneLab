#!/usr/bin/env python3
"""real2sim viewer: reproduce the real cube's pose inside the Genesis sim.

Subscribes to the cube-pose ZMQ feed (published by ``cube_world_observer``), lifts
each tag-frame pose into sim-world coordinates, and places the sim cube there every
frame so the Genesis viewer mirrors reality. Optionally publishes a goal orientation
(drag-free fixed goal) on the goal port for ``play_real``.

Usage:
    # Terminal 1: real camera -> ZMQ (needs hardware; see cube_world_observer.py)
    python -m genelab_wuji.deploy.scripts.cube_world_observer

    # Terminal 2: mirror the cube in the Genesis sim
    python -m genelab_wuji.deploy.scripts.toreal_viewer

Run on a host with a GPU + display (Genesis viewer). The transform math itself is
covered by the headless deploy tests.
"""

from __future__ import annotations

import argparse
import time

from genelab_wuji.deploy.real2sim import cube_pose_in_tag_to_world
from genelab_wuji.deploy.scripts._env import build_reorient_env, set_cube_pose, tag_world_pose
from genelab_wuji.deploy.zmq_bridge import DEFAULT_CUBE_PORT, CubeReceiver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-port", type=int, default=DEFAULT_CUBE_PORT)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--fps", type=float, default=60.0, help="viewer refresh rate")
    args = parser.parse_args()

    env = build_reorient_env()
    tag_pos_w, tag_quat_w = tag_world_pose(env)  # fixed-base hand -> constant
    cube = CubeReceiver(port=args.cube_port, host=args.host)

    print(f"[toreal_viewer] mirroring cube from tcp://{args.host}:{args.cube_port}")
    dt = 1.0 / max(1e-3, args.fps)
    try:
        while not env.viewer_closed:
            cube_pos_tag, cube_quat_tag = cube.latest()
            cube_pos_w, cube_quat_w = cube_pose_in_tag_to_world(
                tag_pos_w, tag_quat_w, cube_pos_tag, cube_quat_tag
            )
            set_cube_pose(env, cube_pos_w, cube_quat_w)
            env.scene.step(update_visualizer=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        cube.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
