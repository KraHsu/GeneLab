#!/usr/bin/env python3
"""Deploy an exported reorient policy to control the (real or mock) Wuji hand.

Wires the tested deploy core into a closed loop:

    cube/goal (ZMQ from cube_world_observer / toreal_viewer)
        -> DeployObsBuilder -> ONNX policy -> EMA action -> hand driver
        -> success monitor (geodesic(cube, goal) < threshold, held) -> resample

Goal modes (``--goal-mode``):
  external  Goal comes from the goal ZMQ feed (toreal_viewer mocap drag). [default]
  fixed     Hold a single goal quat (``--goal-quat w,x,y,z``).
  random    Uniform-SO(3) goal; resampled each time the cube achieves it.

Defaults to ``--mock`` (no hardware, no ZMQ required) so the loop can be smoke-run
anywhere; pass ``--real`` to drive the hand via ``wujihandpy``. The control logic
is covered headlessly by ``tests/test_examples_wuji_deploy_controller.py``.

Usage:
    # Smoke run without hardware (mock hand, random goals):
    python -m genelab_wuji.deploy.scripts.play_real --ckpt policy.onnx --goal-mode random --steps 200

    # Real hand, random goals resampled on success:
    python -m genelab_wuji.deploy.scripts.play_real --ckpt policy.onnx --real --goal-mode random

    # Real hand, goal driven by toreal_viewer over ZMQ:
    python -m genelab_wuji.deploy.scripts.play_real --ckpt policy.onnx --real --goal-mode external
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from genelab_wuji.deploy.config import ENC_TO_POLICY, default_joint_pos_policy
from genelab_wuji.deploy.controller import DeployController
from genelab_wuji.deploy.hand_driver import HandDriverBase, MockHandDriver
from genelab_wuji.deploy.onnx_policy import ONNXPolicy
from genelab_wuji.deploy.zmq_bridge import (
    DEFAULT_CUBE_PORT,
    DEFAULT_GOAL_PORT,
    CubeReceiver,
    GoalReceiver,
)


def _quat_geodesic(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    """Angle (rad) between two unit wxyz quaternions."""
    dot = abs(float(np.dot(q1_wxyz, q2_wxyz)))
    return 2.0 * float(np.arccos(min(1.0, max(-1.0, dot))))


def _random_unit_quat_wxyz() -> np.ndarray:
    """Uniform random rotation over SO(3) (scipy; xyzw -> wxyz)."""
    from scipy.spatial.transform import Rotation

    q_xyzw = Rotation.random().as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def _parse_quat_wxyz(s: str) -> np.ndarray:
    """argparse type: 'w,x,y,z' -> normalized wxyz quaternion."""
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"--goal-quat expects 4 floats w,x,y,z, got {s!r}")
    q = np.array(parts, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        raise argparse.ArgumentTypeError("--goal-quat has zero norm")
    return q / n


class _GoalStub:
    """Local goal source (drop-in for ``GoalReceiver``): ``latest()`` returns the
    current target quat; ``set`` swaps it (used by fixed / random goal modes)."""

    def __init__(self, quat_wxyz: np.ndarray) -> None:
        self._quat = np.asarray(quat_wxyz, dtype=float)

    def set(self, quat_wxyz: np.ndarray) -> None:
        self._quat = np.asarray(quat_wxyz, dtype=float)

    def latest(self) -> np.ndarray:
        return self._quat.copy()

    def close(self) -> None:  # parity with GoalReceiver for the cleanup path
        pass


def _make_driver(real: bool) -> HandDriverBase:
    if not real:
        return MockHandDriver()
    from genelab_wuji.deploy.hand_driver import WujiHandDriver

    driver = WujiHandDriver()
    driver.__enter__()  # caller exits via the finally block in main()
    return driver


def _make_goal_source(args: argparse.Namespace):
    """Return the goal source for the chosen ``--goal-mode``."""
    if args.goal_mode == "fixed":
        if args.goal_quat is None:
            raise SystemExit("--goal-mode fixed requires --goal-quat w,x,y,z")
        return _GoalStub(args.goal_quat)
    if args.goal_mode == "random":
        return _GoalStub(_random_unit_quat_wxyz())
    # external
    if args.no_zmq:
        return GoalReceiver(connect=False)
    return GoalReceiver(port=args.goal_port)


class _SimMirror:
    """Genesis digital-twin viewer for play_real: live hand + observed cube + goal.

    Renders kinematically each control step (no physics — see
    ``InteractiveScene.refresh_visualizer``), so it just shows reality, never fights
    it. All heavy imports (Genesis / the env) are deferred to construction so a
    ``--no-viewer`` run stays numpy-only and headless-safe.
    """

    def __init__(self) -> None:
        from genelab_wuji.deploy.frame_transform import quat_mul
        from genelab_wuji.deploy.real2sim import cube_pose_in_tag_to_world
        from genelab_wuji.deploy.scripts._env import (
            build_reorient_env,
            set_cube_pose,
            set_goal_marker,
            set_hand_joints,
            tag_world_pose,
        )

        self._to_world = cube_pose_in_tag_to_world
        self._quat_mul = quat_mul
        self._set_cube = set_cube_pose
        self._set_goal = set_goal_marker
        self._set_hand = set_hand_joints
        self._env = build_reorient_env(num_envs=1)
        self._tag_pos_w, self._tag_quat_w = tag_world_pose(self._env)

    @property
    def closed(self) -> bool:
        return bool(self._env.viewer_closed)

    def update(
        self,
        joint_pos: np.ndarray,
        cube_pos_tag: np.ndarray,
        cube_quat_tag: np.ndarray,
        goal_quat_tag: np.ndarray,
    ) -> None:
        self._set_hand(self._env, joint_pos)
        cube_pos_w, cube_quat_w = self._to_world(
            self._tag_pos_w, self._tag_quat_w, cube_pos_tag, cube_quat_tag
        )
        self._set_cube(self._env, cube_pos_w, cube_quat_w)
        self._set_goal(self._env, self._quat_mul(self._tag_quat_w, goal_quat_tag))
        self._env.scene.refresh_visualizer()

    def close(self) -> None:
        self._env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="exported policy.onnx")
    parser.add_argument("--metadata", default=None, help="policy metadata.json (auto-detected)")
    parser.add_argument("--real", action="store_true", help="drive the real hand (wujihandpy)")
    parser.add_argument("--mock", action="store_true", help="use the mock hand (default)")
    parser.add_argument("--no-zmq", action="store_true", help="skip ZMQ; zeros cube / stub goal")
    parser.add_argument("--cube-port", type=int, default=DEFAULT_CUBE_PORT)
    parser.add_argument("--goal-port", type=int, default=DEFAULT_GOAL_PORT)
    parser.add_argument("--goal-mode", choices=("external", "fixed", "random"), default="external")
    parser.add_argument(
        "--goal-quat", type=_parse_quat_wxyz, default=None, help="fixed goal 'w,x,y,z'"
    )
    parser.add_argument(
        "--success-threshold", type=float, default=0.2, help="success geodesic err (rad)"
    )
    parser.add_argument(
        "--success-hold-sec", type=float, default=0.5, help="hold time under threshold for success"
    )
    parser.add_argument("--control-dt", type=float, default=0.05, help="policy step period (s)")
    parser.add_argument("--steps", type=int, default=0, help="stop after N steps (0 = forever)")
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mirror the live hand + cube + goal in a Genesis viewer "
        "(default on; pass --no-viewer for headless / mock smoke runs)",
    )
    args = parser.parse_args()

    if args.control_dt <= 0:
        raise SystemExit("--control-dt must be > 0 (used for joint velocity + success timing)")

    policy = ONNXPolicy(args.ckpt, metadata_path=args.metadata)
    driver = _make_driver(real=args.real)
    cube = CubeReceiver(connect=False) if args.no_zmq else CubeReceiver(port=args.cube_port)
    goal = _make_goal_source(args)

    controller = DeployController(
        policy=policy,
        driver=driver,
        cube_source=cube,
        goal_source=goal,
        default_joint_pos=default_joint_pos_policy(),  # policy (articulation) order
        control_dt=args.control_dt,
        enc_to_policy=np.asarray(ENC_TO_POLICY),  # remap encoder<->policy joint order
    )
    controller.reset()
    mirror = _SimMirror() if args.viewer else None

    hold_steps = max(1, round(args.success_hold_sec / args.control_dt))
    print(
        f"[play_real] obs_dim={policy.input_dim} action_dim={policy.action_dim} "
        f"driver={type(driver).__name__} goal_mode={args.goal_mode} "
        f"viewer={'on' if mirror else 'off'} "
        f"success<{args.success_threshold:.2f}rad held {hold_steps} steps"
    )

    step = 0
    hold = 0
    successes = 0
    try:
        while args.steps == 0 or step < args.steps:
            t0 = time.time()
            info = controller.step()
            step += 1

            cube_pos_tag, cube_quat_tag = cube.latest()
            goal_quat_tag = goal.latest()

            # Success monitor: geodesic(cube, goal) below threshold, sustained.
            err = _quat_geodesic(cube_quat_tag, goal_quat_tag)
            hold = hold + 1 if err < args.success_threshold else 0
            if hold >= hold_steps:
                successes += 1
                print(f"[play_real] ✓ success #{successes} (err {np.degrees(err):.1f}°)")
                hold = 0
                if args.goal_mode == "random" and isinstance(goal, _GoalStub):
                    goal.set(_random_unit_quat_wxyz())
                    goal_quat_tag = goal.latest()
                    print("[play_real]   new random goal")

            if mirror is not None:
                mirror.update(info["joint_pos"], cube_pos_tag, cube_quat_tag, goal_quat_tag)
                if mirror.closed:
                    break

            sleep = args.control_dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        driver_exit = getattr(driver, "__exit__", None)
        if args.real and driver_exit is not None:
            driver_exit(None, None, None)
        cube.close()
        goal.close()
        if mirror is not None:
            mirror.close()
    print(f"[play_real] ran {step} control steps, {successes} successes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
