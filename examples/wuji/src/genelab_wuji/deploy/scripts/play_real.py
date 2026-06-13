#!/usr/bin/env python3
"""Deploy an exported reorient policy to control the (real or mock) Wuji hand.

Wires the tested deploy core into a closed loop:

    cube/goal (ZMQ from cube_world_observer/toreal_viewer)
        -> DeployObsBuilder -> ONNX policy -> EMA action -> hand driver

Defaults to ``--mock`` (no hardware, no ZMQ required) so the loop can be smoke-run
anywhere; pass ``--real`` to drive the hand via ``wujihandpy``. The control logic
is covered headlessly by ``tests/test_examples_wuji_deploy_controller.py``.

Usage:
    # Smoke run without hardware (zeros cube/goal -> hand holds the grasp):
    python -m genelab_wuji.deploy.scripts.play_real --ckpt policy.onnx --mock --steps 100

    # Real hand + live observer feed:
    python -m genelab_wuji.deploy.scripts.play_real --ckpt policy.onnx --real
"""

from __future__ import annotations

import argparse
import time

from genelab_wuji.deploy.config import default_joint_pos
from genelab_wuji.deploy.controller import DeployController
from genelab_wuji.deploy.hand_driver import HandDriverBase, MockHandDriver
from genelab_wuji.deploy.onnx_policy import ONNXPolicy
from genelab_wuji.deploy.zmq_bridge import (
    DEFAULT_CUBE_PORT,
    DEFAULT_GOAL_PORT,
    CubeReceiver,
    GoalReceiver,
)


def _make_driver(real: bool) -> HandDriverBase:
    if not real:
        return MockHandDriver()
    from genelab_wuji.deploy.hand_driver import WujiHandDriver

    driver = WujiHandDriver()
    driver.__enter__()  # caller exits via the finally block in main()
    return driver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="exported policy.onnx")
    parser.add_argument("--metadata", default=None, help="policy metadata.json (auto-detected)")
    parser.add_argument("--real", action="store_true", help="drive the real hand (wujihandpy)")
    parser.add_argument("--mock", action="store_true", help="use the mock hand (default)")
    parser.add_argument("--no-zmq", action="store_true", help="skip ZMQ; feed zeros cube/identity goal")
    parser.add_argument("--cube-port", type=int, default=DEFAULT_CUBE_PORT)
    parser.add_argument("--goal-port", type=int, default=DEFAULT_GOAL_PORT)
    parser.add_argument("--control-dt", type=float, default=0.05, help="policy step period (s)")
    parser.add_argument("--steps", type=int, default=0, help="stop after N steps (0 = run forever)")
    args = parser.parse_args()

    policy = ONNXPolicy(args.ckpt, metadata_path=args.metadata)
    driver = _make_driver(real=args.real)

    if args.no_zmq:
        cube = CubeReceiver(connect=False)
        goal = GoalReceiver(connect=False)
    else:
        cube = CubeReceiver(port=args.cube_port)
        goal = GoalReceiver(port=args.goal_port)

    controller = DeployController(
        policy=policy,
        driver=driver,
        cube_source=cube,
        goal_source=goal,
        default_joint_pos=default_joint_pos(),
        control_dt=args.control_dt,
    )
    controller.reset()

    print(f"[play_real] obs_dim={policy.input_dim} action_dim={policy.action_dim} "
          f"driver={type(driver).__name__}")
    step = 0
    try:
        while args.steps == 0 or step < args.steps:
            t0 = time.time()
            controller.step()
            step += 1
            sleep = args.control_dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        if args.real and hasattr(driver, "__exit__"):
            driver.__exit__(None, None, None)
        cube.close()
        goal.close()
    print(f"[play_real] ran {step} control steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
