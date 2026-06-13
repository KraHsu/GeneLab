#!/usr/bin/env python3
"""Vision pipeline: detect the cube pose in the wrist-tag frame and publish on ZMQ.

A simplified, hardware-agnostic port of the wuji-mjlab cube observer. It:

1. grabs camera frames (OpenCV ``VideoCapture`` by default),
2. detects the world AprilTag (ID 0) to fix the wrist-tag frame in the camera,
3. detects the cube's ArUco markers and solves its pose in the camera frame,
4. lifts the cube into the tag frame via the tested :func:`cube_cam_to_tag`, and
5. publishes ``(pos, quat_wxyz, world_fixed, cube_size)`` via :class:`CubePublisher`.

The frame math (steps 4-5) is covered by the headless deploy tests; the detector
(steps 1-3) needs a real camera + ``opencv-contrib-python`` + ``pupil-apriltags``
and is not exercised in CI. For the production Hikvision rig, swap ``_open_camera``
and reuse the rest. Filtering (Kalman / dominant-face disambiguation) from the
reference is intentionally omitted here for clarity.

Usage:
    python -m genelab_wuji.deploy.scripts.cube_world_observer --camera 0
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from genelab_wuji.deploy.frame_transform import cube_cam_to_tag
from genelab_wuji.deploy.zmq_bridge import DEFAULT_CUBE_PORT, CubePublisher

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
WORLD_TAG_ID = 0
WORLD_TAG_SIZE = 0.048  # meters (48 mm wrist AprilTag)


def _camera_matrix(cam_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    k = cam_cfg["intrinsics"]
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], dtype=float)
    d = cam_cfg["distortion"]
    dist = np.array([d["k1"], d["k2"], d["p1"], d["p2"], d["k3"]], dtype=float)
    return K, dist


def _open_camera(index: int):
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {index}")
    return cap


def main() -> int:
    import cv2
    from pupil_apriltags import Detector

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cube-port", type=int, default=DEFAULT_CUBE_PORT)
    parser.add_argument("--camera-config", default=os.path.join(_CONFIG_DIR, "camera.yaml"))
    parser.add_argument("--cube-config", default=os.path.join(_CONFIG_DIR, "cube_tags.json"))
    args = parser.parse_args()

    with open(args.camera_config) as f:
        cam_cfg = yaml.safe_load(f)
    K, dist = _camera_matrix(cam_cfg)

    import json

    with open(args.cube_config) as f:
        cube_cfg = json.load(f)
    cube_size = float(cube_cfg["cube_size"])

    cap = _open_camera(args.camera)
    apriltag = Detector(families="tag36h11")
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    publisher = CubePublisher(port=args.cube_port)

    world_pose: tuple[np.ndarray, np.ndarray] | None = None  # (R_tag_cam, t_tag_cam)
    print(f"[cube_world_observer] publishing on tcp://*:{args.cube_port}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1) World/wrist tag -> fix the tag frame in the camera.
            cam_params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            for r in apriltag.detect(
                gray, estimate_tag_pose=True, camera_params=cam_params, tag_size=WORLD_TAG_SIZE
            ):
                if r.tag_id == WORLD_TAG_ID:
                    world_pose = (np.asarray(r.pose_R), np.asarray(r.pose_t).flatten())

            # 2) Cube ArUco -> pose in camera frame (single-marker fallback PnP).
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            cube_cam = _solve_cube_pose(corners, ids, cube_cfg, K, dist)

            if world_pose is not None and cube_cam is not None:
                R_cube_cam, t_cube_cam = cube_cam
                R_cube_tag, t_cube_tag = cube_cam_to_tag(
                    world_pose[0], world_pose[1], R_cube_cam, t_cube_cam
                )
                quat_xyzw = Rotation.from_matrix(R_cube_tag).as_quat()
                quat_wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]])
                publisher.publish(
                    t_cube_tag, quat_wxyz, world_fixed=True, cube_size=cube_size
                )
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.close()
        cap.release()
    return 0


def _solve_cube_pose(corners, ids, cube_cfg, K, dist):
    """Estimate the cube pose in the camera frame from its visible ArUco markers.

    Simplified single-marker estimate: take the first detected face marker, recover
    its pose, and offset by half the cube to the cube center along the face normal.
    The reference observer fuses all visible faces with a dominant-face strategy.
    """
    import cv2

    if ids is None or len(ids) == 0:
        return None
    tag_size = float(cube_cfg["tag_size"])
    half = float(cube_cfg["cube_size"]) / 2.0
    obj = np.array(
        [[-tag_size / 2, tag_size / 2, 0], [tag_size / 2, tag_size / 2, 0],
         [tag_size / 2, -tag_size / 2, 0], [-tag_size / 2, -tag_size / 2, 0]],
        dtype=np.float32,
    )
    ok, rvec, tvec = cv2.solvePnP(obj, corners[0][0], K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten() + R @ np.array([0.0, 0.0, -half])  # face -> cube center
    return R, t


if __name__ == "__main__":
    raise SystemExit(main())
