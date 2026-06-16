#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""
Cube World Observer

Detects cube pose relative to world coordinate system defined by AprilTag.
Publishes pose via ZMQ for use by play_real.py.

Ported into GeneLab from wuji-mjlab deploy/reorient/scripts/cube_world_observer.py.

Features:
- World frame defined by AprilTag ID 0
- Cube detection using ArUco 4x4 tags with dominant face strategy
- SO3 Kalman Filter for rotation smoothing
- ZMQ publishing on port 5555

Usage:
    python -m genelab_wuji.deploy.scripts.cube_world_observer --preview  # With visualization
    python -m genelab_wuji.deploy.scripts.cube_world_observer            # Headless mode

On startup, the world coordinate system is auto-sampled (100 frames by default),
then a fixed world frame is used. Press 'w' to resample the world frame.
"""
import sys
import os
import time
import json
import yaml
import numpy as np
import cv2
from ctypes import *
from scipy.spatial.transform import Rotation
from scipy.linalg import inv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # deploy/scripts
ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # deploy
# MvImport: Hikvision MVS SDK Python bindings.
# System-level dependency (NOT vendored in this repo). Default install path is
# /opt/MVS; override with MVS_PYTHON_PATH env var if installed elsewhere.
_mvs_python_path = os.environ.get("MVS_PYTHON_PATH", "/opt/MVS/Samples/64/Python")
if not os.path.isdir(os.path.join(_mvs_python_path, "MvImport")):
    raise RuntimeError(
        f"MvImport not found at {_mvs_python_path}/MvImport. "
        "Install Hikvision MVS SDK (https://www.hikrobotics.com) or set "
        "MVS_PYTHON_PATH env var to the dir containing MvImport/."
    )
sys.path.insert(0, _mvs_python_path)

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

from MvImport.MvCameraControl_class import *
from genelab_wuji.deploy.camera_config import (
    load_camera_config, get_camera_matrix, get_dist_coeffs,
    setup_camera_roi, setup_camera_capture
)
from genelab_wuji.deploy.zmq_bridge import DEFAULT_CUBE_PORT, CubePublisher

try:
    from pupil_apriltags import Detector as AprilTagDetector
except ImportError:
    print("ERROR: pupil_apriltags not installed. Run: pip install pupil-apriltags")
    sys.exit(1)

import zmq

# Load camera config
_cam_cfg = load_camera_config()
K = get_camera_matrix(_cam_cfg)
DIST_COEFFS = get_dist_coeffs(_cam_cfg)

# Config files
OBSERVER_CONFIG_FILE = os.path.join(os.path.dirname(SCRIPT_DIR), "config", "observer.yaml")  # deploy/config/observer.yaml

from genelab_wuji.deploy.cube_geom import (
    resolve_cube_config_path,
    DEFAULT_CUBE_CONFIG_FILE as CUBE_CONFIG_FILE,
)

# World origin = AprilTag ID 0 on the wrist.
WORLD_TAG_ID = 0
WORLD_TAG_SIZE = 0.048  # 48mm
WORLD_SAMPLE_FRAMES = 100  # Number of frames to sample for world frame averaging

# Optional world-frame correction; None = use AprilTag frame as-is.
# AprilTag detector (X-right, Y-down, Z-into-tag) -> MuJoCo wrist tag (right-handed)
# Pure handedness flip: same X, flipped Y and Z (printed tag X aligns with MuJoCo wrist tag X).
WORLD_FRAME_CORRECTION = np.array([
    [ 1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
    [ 0.0,  0.0, -1.0],
])
# WORLD_FRAME_CORRECTION = "+x +z -y"  # Example: remap axes
# WORLD_FRAME_CORRECTION = np.array([[1,0,0], [0,0,1], [0,-1,0]])  # Same as above


def parse_axis_remap(remap_str):
    """Parse axis remapping string to rotation matrix.

    Args:
        remap_str: String like "+x +z -y" specifying how AprilTag axes map to new world axes
                   Format: "new_X new_Y new_Z" where each is ±x, ±y, or ±z

    Returns:
        3x3 rotation matrix R such that new_point = R @ apriltag_point
    """
    axis_map = {
        '+x': np.array([1, 0, 0]),
        '-x': np.array([-1, 0, 0]),
        '+y': np.array([0, 1, 0]),
        '-y': np.array([0, -1, 0]),
        '+z': np.array([0, 0, 1]),
        '-z': np.array([0, 0, -1]),
    }

    parts = remap_str.lower().split()
    if len(parts) != 3:
        raise ValueError(f"Axis remap must have 3 parts, got: {remap_str}")

    # Build rotation matrix: columns are where AprilTag axes go
    # But we want rows to be where new axes come from
    R = np.zeros((3, 3))
    for i, part in enumerate(parts):
        if part not in axis_map:
            raise ValueError(f"Invalid axis: {part}. Use +x,-x,+y,-y,+z,-z")
        R[i, :] = axis_map[part]

    # Verify it's a valid rotation (det = +1 for right-handed)
    det = np.linalg.det(R)
    if not np.isclose(abs(det), 1.0):
        raise ValueError(f"Invalid axis remap: axes not orthogonal (det={det:.3f})")
    if det < 0:
        raise ValueError(f"Invalid axis remap: forms left-handed system (det={det:.3f}). "
                        "Hint: flip one axis sign to make it right-handed.")

    return R

# No silent default — pass --cube to override config/cube_tags.json.

# Cube frame correction rotation matrix
# Corrects the difference between ArUco board coordinate system and MuJoCo mesh coordinate system
# Format: same as WORLD_FRAME_CORRECTION (matrix or axis remap string)
# Set to None if ArUco board axes match MuJoCo mesh axes
CUBE_FRAME_CORRECTION = None  # None = no correction
# Example: if ArUco X,Y,Z maps to MuJoCo Y,Z,X: CUBE_FRAME_CORRECTION = "+y +z +x"

# Face colors (matching MuJoCo dex_cube)
FACE_COLORS = {
    'TOP':    ('Cyan',   (255, 255, 0)),    # BGR
    'BOTTOM': ('Blue',   (255, 0, 0)),
    'FRONT':  ('Red',    (0, 0, 255)),
    'BACK':   ('White',  (255, 255, 255)),
    'LEFT':   ('Green',  (0, 255, 0)),
    'RIGHT':  ('Yellow', (0, 255, 255)),
}


def load_observer_config():
    """Load observer configuration from YAML file."""
    defaults = {
        'rotation_filter': {
            'process_noise': 0.1,
            'measurement_noise': 0.3,
        },
        'position_filter': {
            'alpha': 0.6,
        },
        'pnp': {
            'reproj_threshold': 6.0,
        },
        'preprocess': {
            'enable_clahe': True,
            'clahe_clip': 2.0,
            'clahe_tile': [8, 8],
        },
    }

    if os.path.exists(OBSERVER_CONFIG_FILE):
        try:
            with open(OBSERVER_CONFIG_FILE, 'r') as f:
                cfg = yaml.safe_load(f)
            # Merge with defaults
            for key in defaults:
                if key in cfg:
                    defaults[key].update(cfg[key])
            print(f"Loaded observer config from {OBSERVER_CONFIG_FILE}")
        except Exception as e:
            print(f"Warning: Failed to load observer config: {e}, using defaults")

    return defaults


class SO3KalmanFilter:
    """SO(3) rotation Kalman filter in tangent space."""

    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.state = np.zeros(3)
        self.covariance = np.eye(3) * 0.1
        self.Q = np.eye(3) * process_noise
        self.R_noise = np.eye(3) * measurement_noise
        self.is_initialized = False
        self.reference_rot = np.eye(3)
        self.filtered_rot = np.eye(3)

    def _rotation_to_axis_angle(self, R):
        """Convert rotation matrix to axis-angle (more stable than logm near 180°)."""
        # Use Rodrigues formula for stability
        rvec, _ = cv2.Rodrigues(R)
        return rvec.flatten()

    def _axis_angle_to_rotation(self, rvec):
        """Convert axis-angle to rotation matrix."""
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        return R

    def update(self, rotation_matrix):
        if not self.is_initialized:
            self.reference_rot = rotation_matrix.copy()
            self.filtered_rot = rotation_matrix.copy()
            self.is_initialized = True
            return rotation_matrix

        # Compute relative rotation
        R_relative = rotation_matrix @ self.reference_rot.T

        # Use axis-angle instead of logm for stability near 180°
        z_local = self._rotation_to_axis_angle(R_relative)

        # Kalman update
        self.covariance = self.covariance + self.Q
        S = self.covariance + self.R_noise
        K_gain = self.covariance @ inv(S)
        self.state = self.state + K_gain @ (z_local - self.state)
        self.covariance = (np.eye(3) - K_gain) @ self.covariance

        # Convert back to rotation matrix
        R_filtered_local = self._axis_angle_to_rotation(self.state)
        R_filtered_global = R_filtered_local @ self.reference_rot

        # Re-center reference when state gets too large
        if np.linalg.norm(self.state) > 1.5:
            self.reference_rot = R_filtered_global.copy()
            self.state = np.zeros(3)

        self.filtered_rot = R_filtered_global
        return self.filtered_rot

    def reset(self):
        self.state = np.zeros(3)
        self.covariance = np.eye(3) * 0.1
        self.is_initialized = False
        self.reference_rot = np.eye(3)


class VectorLowPassFilter:
    """Simple low-pass filter for position."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.filtered_val = None

    def update(self, val):
        if self.filtered_val is None:
            self.filtered_val = val.copy()
            return self.filtered_val
        self.filtered_val = self.alpha * val + (1 - self.alpha) * self.filtered_val
        return self.filtered_val

    def reset(self):
        self.filtered_val = None


# Corner EMA filter alpha (1.0 = no smoothing, acts as per-ID state cache for reset)
CORNER_FILTER_ALPHA = 1.0


class CornerEMAFilter:
    """Per-marker-ID corner EMA filter.

    Maintains a dict {marker_id: (4,2) filtered corners}.
    Each frame, for every detected marker, the 4 corner positions are
    exponentially-smoothed with the previous frame's value.
    Markers not seen for >max_age frames are evicted.
    """

    def __init__(self, alpha=CORNER_FILTER_ALPHA, max_age=5):
        self.alpha = alpha
        self.max_age = max_age
        self._state = {}
        self._age = {}

    def update(self, corners, ids):
        """Filter corners in-place and return (filtered_corners, ids)."""
        if ids is None or len(ids) == 0:
            for mid in list(self._age):
                self._age[mid] += 1
                if self._age[mid] > self.max_age:
                    del self._state[mid]
                    del self._age[mid]
            return corners, ids

        seen = set()
        filtered = []
        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)
            seen.add(mid)
            pts = corners[i].reshape(4, 2).astype(np.float32)
            if mid in self._state:
                pts = self.alpha * pts + (1 - self.alpha) * self._state[mid]
            self._state[mid] = pts.copy()
            self._age[mid] = 0
            filtered.append(pts.reshape(1, 4, 2).astype(np.float32))

        for mid in list(self._age):
            if mid not in seen:
                self._age[mid] += 1
                if self._age[mid] > self.max_age:
                    del self._state[mid]
                    del self._age[mid]

        return filtered, ids

    def reset(self):
        self._state.clear()
        self._age.clear()


# --- Buffer backlog detection constants ---
BACKLOG_LATENCY_S = 30.0e-3     # 30ms; headless grab ≈ 20ms (waiting for camera frame)
BACKLOG_COUNT = 5                # consecutive slow grabs before flush
BACKLOG_MAX_FLUSH = 20           # safety cap on flush loop


class CubeWorldObserver:
    """Detects cube pose relative to world coordinate system defined by AprilTag."""

    def __init__(self, visualize=False, zmq_port=5555,
                 process_noise=0.01, measurement_noise=1.0, alpha=0.3,
                 world_sample_frames=WORLD_SAMPLE_FRAMES,
                 cube_config_path: str | None = None):
        self.visualize = visualize
        self._cube_config_path = cube_config_path or CUBE_CONFIG_FILE

        # Initialize camera
        print("Initializing camera...")
        MvCamera.MV_CC_Initialize()
        deviceList = MV_CC_DEVICE_INFO_LIST()
        MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)

        if deviceList.nDeviceNum == 0:
            raise RuntimeError("No camera found!")

        self.cam = MvCamera()
        stDevice = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        self.cam.MV_CC_CreateHandle(stDevice)
        self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerGB8)
        setup_camera_capture(self.cam, _cam_cfg)
        setup_camera_roi(self.cam, _cam_cfg)
        self.cam.MV_CC_StartGrabbing()
        print("Camera ready!")

        # AprilTag detector for world frame
        self.apriltag_detector = AprilTagDetector(
            families="tag36h11", nthreads=4, quad_decimate=1.0,
            quad_sigma=0.0, decode_sharpening=0.25,
        )

        # ArUco detector for cube
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # Support both old and new OpenCV API
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            # Old API (OpenCV < 4.7)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.aruco_detector = None  # Use old-style detectMarkers
        else:
            # New API (OpenCV >= 4.7)
            aruco_params = cv2.aruco.DetectorParameters()
            aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, aruco_params)

        # Load cube config and build board
        self._load_config()
        self._build_aruco_board()

        # Filters
        self.filter_R = SO3KalmanFilter(process_noise=process_noise, measurement_noise=measurement_noise)
        self.filter_t = VectorLowPassFilter(alpha=alpha)

        # ZMQ publisher
        self.publisher = CubePublisher(port=zmq_port)
        print(f"ZMQ publisher on port {zmq_port}")

        # State
        self.stOutFrame = MV_FRAME_OUT()
        self.world_pose = None
        self.frame_count = 0
        self.last_print_time = 0
        self.last_frame_count = 0
        self._display_fps = 0.0
        self.filt_R = np.eye(3)
        self.filt_t = np.zeros(3)
        self.prev_quat = None  # For quaternion sign continuity
        self._R_cube_world = None  # Cube rotation in world frame (for visualization)
        self._t_cube_world = None  # Cube position in world frame (for visualization)
        self._dominant_face = None  # Currently visible face

        # World frame sampling state
        self._world_samples_R = []  # Collected rotation samples
        self._world_samples_t = []  # Collected translation samples
        self._world_fixed = False   # Whether world frame is fixed
        self._world_sample_target = world_sample_frames  # Target sample count

        # --- New observer state for IPPE+ITERATIVE migration ---
        _cfg = load_observer_config()
        self._reproj_threshold = float(_cfg['pnp']['reproj_threshold'])

        self._enable_clahe = bool(_cfg['preprocess']['enable_clahe'])
        _clip = float(_cfg['preprocess']['clahe_clip'])
        _tile = tuple(int(x) for x in _cfg['preprocess']['clahe_tile'])
        self._clahe = cv2.createCLAHE(clipLimit=_clip, tileGridSize=_tile) if self._enable_clahe else None

        self.corner_filter = CornerEMAFilter(alpha=CORNER_FILTER_ALPHA)

        # IPPE disambiguation state
        self._ippe_locked_idx = 0
        self._lost_frames = 0
        self._prev_dominant_face = None
        self._active_faces = set()
        self._reproj_err = 0.0

        # Backlog detection state
        self._grab_slow_count = 0

    def _load_config(self):
        """Load cube configuration from file.

        Raises:
            FileNotFoundError: if the cube_tags*.json path does not exist —
                cube geometry must be specified explicitly (no silent defaults).
            KeyError: if the JSON is missing required cube_size/tag_size keys.
        """
        self._tag_map = None
        self._face_axes_cfg = None
        self._face_rotations = {'TOP': 0, 'BOTTOM': 0, 'FRONT': 0, 'BACK': 0, 'LEFT': 0, 'RIGHT': 0}

        if not os.path.exists(self._cube_config_path):
            raise FileNotFoundError(
                f"cube tags JSON not found: {self._cube_config_path}. "
                "Specify with --cube (e.g. --cube 36 / --cube 40_5 / default 54mm)."
            )

        try:
            with open(self._cube_config_path, 'r') as f:
                cfg = json.load(f)
            # Required: cube_size + tag_size + tag_center_offset. No silent defaults.
            try:
                self._cube_size = float(cfg['cube_size'])
                self._tag_size = float(cfg['tag_size'])
                self._tag_offset = float(cfg['tag_center_offset'])
            except KeyError as e:
                raise KeyError(
                    f"{self._cube_config_path} is missing required key {e}; "
                    "cube_size, tag_size and tag_center_offset are not allowed to be defaulted."
                ) from None
            faces_cfg = cfg.get('faces_config', {})
            self._tag_map = {face: {int(k): v for k, v in tags.items()} for face, tags in faces_cfg.items()}
            self._face_axes_cfg = cfg.get('face_axes', None)
            for face, rot in cfg.get('face_rotations', {}).items():
                self._face_rotations[face] = rot
            print(f"Loaded cube config from {self._cube_config_path}")
            print(f"  cube_size={self._cube_size*1000:.1f}mm  "
                  f"tag_size={self._tag_size*1000:.2f}mm  "
                  f"tag_center_offset={self._tag_offset*1000:.2f}mm")
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse config JSON: {e}")

        # Build tag to face mapping
        self._tag_to_face = {}
        if self._tag_map:
            for face, tags in self._tag_map.items():
                for tid in tags.keys():
                    self._tag_to_face[tid] = face

    def _build_aruco_board(self):
        """Build ArUco Board with all cube tags' 3D positions."""
        half = self._cube_size / 2
        ht = self._tag_size / 2
        off = self._tag_offset

        def rotate_corners(corners, rotation):
            n = (rotation // 90) % 4
            if n == 0: return corners
            elif n == 1: return np.array([corners[3], corners[0], corners[1], corners[2]])
            elif n == 2: return np.array([corners[2], corners[3], corners[0], corners[1]])
            else: return np.array([corners[1], corners[2], corners[3], corners[0]])

        def face_tags(face_center, u_axis, v_axis, rotation=0):
            tags = {}
            for pos, center in [('T', face_center + off * v_axis), ('B', face_center - off * v_axis),
                               ('L', face_center - off * u_axis), ('R', face_center + off * u_axis)]:
                corners = np.array([
                    center - ht * u_axis + ht * v_axis, center + ht * u_axis + ht * v_axis,
                    center + ht * u_axis - ht * v_axis, center - ht * u_axis - ht * v_axis,
                ], dtype=np.float32)
                tags[pos] = rotate_corners(corners, rotation)
            return tags

        if self._face_axes_cfg:
            faces = {name: (np.array(axes['center'], dtype=np.float64) * half,
                           np.array(axes['u'], dtype=np.float64),
                           np.array(axes['v'], dtype=np.float64))
                    for name, axes in self._face_axes_cfg.items()}
        else:
            X, Y, Z = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
            faces = {'TOP': (half*Z, X, Y), 'BOTTOM': (-half*Z, X, -Y), 'FRONT': (-half*Y, X, Z),
                    'BACK': (half*Y, -X, Z), 'LEFT': (-half*X, -Y, Z), 'RIGHT': (half*X, Y, Z)}

        tag_map = self._tag_map or {
            'TOP': {0:'L',1:'B',2:'T',3:'R'}, 'BOTTOM': {8:'R',9:'T',10:'B',11:'L'},
            'FRONT': {16:'R',17:'T',18:'B',19:'L'}, 'BACK': {20:'B',21:'R',22:'L',23:'T'},
            'LEFT': {4:'R',5:'T',6:'B',7:'L'}, 'RIGHT': {12:'B',13:'R',14:'L',15:'T'},
        }

        board_corners, board_ids = [], []
        for face_name, (center, u, v) in faces.items():
            tags = face_tags(center, u, v, self._face_rotations.get(face_name, 0))
            for tid, pos in tag_map[face_name].items():
                board_corners.append(tags[pos])
                board_ids.append([tid])

        sorted_idx = np.argsort([b[0] for b in board_ids])
        board_corners = [board_corners[i] for i in sorted_idx]
        board_ids = np.array([board_ids[i] for i in sorted_idx], dtype=np.int32)

        # Support both old and new OpenCV API
        if hasattr(cv2.aruco, 'Board_create'):
            # Old API (OpenCV < 4.7)
            self.cube_board = cv2.aruco.Board_create(board_corners, self.aruco_dict, board_ids)
        else:
            # New API (OpenCV >= 4.7)
            self.cube_board = cv2.aruco.Board(board_corners, self.aruco_dict, board_ids)
        print(f"ArUco Board: {len(board_ids)} tags")

    def _match_image_points(self, corners, ids):
        """Match detected corners/ids to board - compatibility wrapper for old/new API."""
        if hasattr(self.cube_board, 'matchImagePoints'):
            # New API (OpenCV >= 4.7)
            return self.cube_board.matchImagePoints(corners, ids)
        else:
            # Old API (OpenCV < 4.7) - manually match
            obj_pts = []
            img_pts = []
            if ids is None or len(ids) == 0:
                return None, None

            board_ids_flat = self.cube_board.ids.flatten()
            for i, marker_id in enumerate(ids.flatten()):
                # Find this marker in the board
                board_idx = np.where(board_ids_flat == marker_id)[0]
                if len(board_idx) > 0:
                    board_idx = board_idx[0]
                    # Add all 4 corners of this marker
                    obj_pts.extend(self.cube_board.objPoints[board_idx])
                    img_pts.extend(corners[i][0])

            if len(obj_pts) == 0:
                return None, None

            return np.array(obj_pts, dtype=np.float32), np.array(img_pts, dtype=np.float32)

    def detect_world_tag(self, gray):
        """Detect world AprilTag and return its pose in camera frame."""
        results = self.apriltag_detector.detect(
            gray, estimate_tag_pose=True,
            camera_params=(K[0, 0], K[1, 1], K[0, 2], K[1, 2]),
            tag_size=WORLD_TAG_SIZE
        )
        for r in results:
            if r.tag_id == WORLD_TAG_ID:
                return r.pose_R, r.pose_t.flatten(), r.corners
        return None, None, None

    def _average_rotations(self, rotations):
        """Average multiple rotation matrices using quaternion averaging."""
        if len(rotations) == 0:
            return np.eye(3)

        # Convert to quaternions
        quats = []
        for R in rotations:
            rot = Rotation.from_matrix(R)
            q = rot.as_quat()  # (x, y, z, w)
            quats.append(q)

        quats = np.array(quats)

        # Ensure quaternion sign consistency (all pointing same hemisphere)
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[0]) < 0:
                quats[i] = -quats[i]

        # Average quaternions and normalize
        avg_quat = quats.mean(axis=0)
        avg_quat /= np.linalg.norm(avg_quat)

        return Rotation.from_quat(avg_quat).as_matrix()

    def start_world_sampling(self):
        """Start/restart world frame sampling."""
        self._world_samples_R = []
        self._world_samples_t = []
        self._world_fixed = False
        self.world_pose = None
        print(f"\n[World Sampling] Starting... (collecting {self._world_sample_target} frames)")

    def _finalize_world_frame(self):
        """Finalize world frame from collected samples."""
        if len(self._world_samples_R) < 10:
            print(f"[World Sampling] Failed: only {len(self._world_samples_R)} samples collected")
            return False

        # Average rotation matrices
        avg_R = self._average_rotations(self._world_samples_R)

        # Average translations
        avg_t = np.mean(self._world_samples_t, axis=0)

        # Apply world frame correction if specified
        if WORLD_FRAME_CORRECTION is not None:
            # Parse string format or use matrix directly
            if isinstance(WORLD_FRAME_CORRECTION, str):
                correction_R = parse_axis_remap(WORLD_FRAME_CORRECTION)
                print(f"[World Sampling] Axis remap: {WORLD_FRAME_CORRECTION}")
            else:
                correction_R = np.array(WORLD_FRAME_CORRECTION)

            # R_corrected transforms points from corrected world frame to camera frame
            # If R_apriltag transforms from AprilTag frame to camera frame,
            # and correction_R transforms from AprilTag frame to new world frame,
            # then: R_new_to_cam = R_apriltag @ correction_R.T
            avg_R = avg_R @ correction_R.T
            print(f"[World Sampling] Applied world frame correction (det={np.linalg.det(correction_R):.1f})")

        self.world_pose = (avg_R, avg_t)
        self._world_fixed = True

        print(f"[World Sampling] Complete! Averaged {len(self._world_samples_R)} samples")
        print(f"[World Sampling] World frame is now FIXED. Press 'w' to resample.")

        # Switch to hardware fast ROI (headless only; preview keeps full frame)
        if not self.visualize:
            self._switch_to_fast_roi()

        return True

    def _switch_to_fast_roi(self):
        """Switch camera to hardware fast_roi for high-speed cube tracking (headless)."""
        global K
        fast_roi = _cam_cfg.get('fast_roi')
        if fast_roi is None:
            return
        cur_roi = _cam_cfg['roi']
        if (fast_roi['width'] == cur_roi['width']
                and fast_roi['height'] == cur_roi['height']
                and fast_roi['offset_x'] == cur_roi['offset_x']
                and fast_roi['offset_y'] == cur_roi['offset_y']):
            return  # already at fast ROI

        print(f"[Fast ROI] Switching to {fast_roi['width']}x{fast_roi['height']} "
              f"@ ({fast_roi['offset_x']}, {fast_roi['offset_y']}) ...")
        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_SetIntValueEx("OffsetX", 0)
        self.cam.MV_CC_SetIntValueEx("OffsetY", 0)
        self.cam.MV_CC_SetIntValueEx("Width", fast_roi['width'])
        self.cam.MV_CC_SetIntValueEx("Height", fast_roi['height'])
        self.cam.MV_CC_SetIntValueEx("OffsetX", fast_roi['offset_x'])
        self.cam.MV_CC_SetIntValueEx("OffsetY", fast_roi['offset_y'])
        self.cam.MV_CC_StartGrabbing()

        # Update global K for new ROI offset
        intr = _cam_cfg['intrinsics']
        K[0, 2] = intr['cx'] - fast_roi['offset_x']
        K[1, 2] = intr['cy'] - fast_roi['offset_y']
        print(f"[Fast ROI] Active. K updated: cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    def detect_cube_pose(self, corners, ids):
        """Detect cube pose via IPPE + ITERATIVE hybrid with dominant-face strategy.

        Pipeline:
          1. Dominant-face selection with hysteresis.
          2. IPPE (coplanar analytical) returns two candidate solutions.
          3. Disambiguate with locked-index hysteresis: switch only if other
             solution is strictly better on BOTH reproj and geodesic distance.
          4. ITERATIVE refinement using the chosen IPPE solution as guess.
          5. Reprojection-error gate; filter reset on reacquire.
        """
        if ids is None or len(ids) == 0:
            self._dominant_face = None
            self._lost_frames += 1
            return None, None, 0

        # --- Count markers per face ---
        face_counts = {}
        for tid in ids.flatten():
            if int(tid) in self._tag_to_face:
                face = self._tag_to_face[int(tid)]
                face_counts[face] = face_counts.get(face, 0) + 1

        if not face_counts:
            self._dominant_face = None
            self._lost_frames += 1
            return None, None, 0

        # --- Dominant face with hysteresis ---
        best_face = max(face_counts, key=face_counts.get)
        if (self._prev_dominant_face is not None
                and self._prev_dominant_face in face_counts
                and face_counts.get(self._prev_dominant_face, 0) >= face_counts[best_face]):
            best_face = self._prev_dominant_face
        self._dominant_face = best_face
        self._prev_dominant_face = best_face
        self._active_faces = {best_face}

        valid_indices = [i for i, tid in enumerate(ids.flatten())
                         if int(tid) in self._tag_to_face and self._tag_to_face[int(tid)] == best_face]
        if valid_indices:
            corners = [corners[i] for i in valid_indices]
            ids = ids[valid_indices]

        obj_pts, img_pts = self._match_image_points(corners, ids)
        if obj_pts is None or len(obj_pts) < 4:
            self._lost_frames += 1
            return None, None, 0

        # --- Step 1: IPPE returns both coplanar solutions (sol 0 has lower reproj) ---
        n_sol, rvecs_ippe, tvecs_ippe, reproj_errors = cv2.solvePnPGeneric(
            obj_pts, img_pts, K, DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE)

        if n_sol == 0:
            self._lost_frames += 1
            return None, None, 0

        # --- Step 2: Disambiguate IPPE solutions ---
        # Lock onto current pick; switch only if other is clearly better
        # on BOTH reproj (<80%) AND geodesic distance (<33%).
        if n_sol == 1:
            best_idx = 0
        elif not self.filter_R.is_initialized or self._lost_frames > 0:
            best_idx = 0
        else:
            R_prev = self.filt_R
            dists = []
            for i in range(n_sol):
                R_i, _ = cv2.Rodrigues(rvecs_ippe[i])
                diff = cv2.Rodrigues(R_prev.T @ R_i)[0]
                dists.append(np.linalg.norm(diff))

            locked = self._ippe_locked_idx
            other = 1 - locked
            re_locked = reproj_errors[locked].item()
            re_other = reproj_errors[other].item()

            if (re_other < re_locked * 0.8) and (dists[other] < dists[locked] * 0.33):
                best_idx = other
            else:
                best_idx = locked

        self._ippe_locked_idx = best_idx
        pick_rvec, pick_tvec = rvecs_ippe[best_idx], tvecs_ippe[best_idx]

        # --- Step 3: ITERATIVE refinement with IPPE pick as initial guess ---
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, DIST_COEFFS,
            rvec=pick_rvec.copy(), tvec=pick_tvec.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            self._lost_frames += 1
            return None, None, 0

        # --- Step 4: Reprojection-error gate ---
        reproj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, DIST_COEFFS)
        reproj_err = float(np.mean(np.linalg.norm(
            img_pts.reshape(-1, 2) - reproj_pts.reshape(-1, 2), axis=1)))
        self._reproj_err = reproj_err
        if reproj_err > self._reproj_threshold:
            self._lost_frames += 1
            return None, None, 0

        # --- Step 5: Reset filters on reacquire so stale state doesn't contaminate ---
        if self._lost_frames > 0:
            self.corner_filter.reset()
            self.filter_R.reset()
            self.filter_t.reset()
            self.prev_quat = None

        self._lost_frames = 0

        # --- Update filters ---
        R, _ = cv2.Rodrigues(rvec)
        self.filt_R = self.filter_R.update(R)
        self.filt_t = self.filter_t.update(tvec.flatten())

        return self.filt_R, self.filt_t, len(ids)

    def transform_to_world_frame(self, R_cube_cam, t_cube_cam):
        """Transform cube pose from camera frame to world frame."""
        if self.world_pose is None:
            return None, None
        R_world_cam, t_world_cam = self.world_pose
        R_cam_world = R_world_cam.T
        t_cam_world = -R_cam_world @ t_world_cam
        R_cube_world = R_cam_world @ R_cube_cam
        t_cube_world = R_cam_world @ t_cube_cam + t_cam_world
        return R_cube_world, t_cube_world

    def _draw_world_axes(self, img, axis_length=0.03, line_width=4):
        """Draw world coordinate axes with RGB colors for XYZ."""
        if self.world_pose is None:
            return

        R_world, t_world = self.world_pose

        # Project origin and axis endpoints to image
        origin = t_world.reshape(3, 1)
        x_end = origin + axis_length * R_world[:, 0:1]
        y_end = origin + axis_length * R_world[:, 1:2]
        z_end = origin + axis_length * R_world[:, 2:3]

        # Project to 2D
        origin_2d, _ = cv2.projectPoints(origin.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)
        x_2d, _ = cv2.projectPoints(x_end.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)
        y_2d, _ = cv2.projectPoints(y_end.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)
        z_2d, _ = cv2.projectPoints(z_end.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)

        origin_pt = tuple(origin_2d[0, 0].astype(int))
        x_pt = tuple(x_2d[0, 0].astype(int))
        y_pt = tuple(y_2d[0, 0].astype(int))
        z_pt = tuple(z_2d[0, 0].astype(int))

        # Draw axes: X=Red, Y=Green, Z=Blue (BGR format)
        cv2.arrowedLine(img, origin_pt, x_pt, (0, 0, 255), line_width, tipLength=0.3)   # X - Red
        cv2.arrowedLine(img, origin_pt, y_pt, (0, 255, 0), line_width, tipLength=0.3)   # Y - Green
        cv2.arrowedLine(img, origin_pt, z_pt, (255, 0, 0), line_width, tipLength=0.3)   # Z - Blue

        # Draw axis labels at arrow tips
        cv2.putText(img, "+X", x_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, "+Y", y_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, "+Z", z_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw origin marker
        cv2.circle(img, origin_pt, 5, (255, 255, 255), -1)

    def _draw_cube_axes_in_world(self, img, R_cube_world, t_cube_world, axis_length=0.025, line_width=2):
        """Draw cube coordinate axes in world frame (transformed to camera for display)."""
        if self.world_pose is None:
            return

        R_world_cam, t_world_cam = self.world_pose

        # self.world_pose = (R_world_cam, t_world_cam) where:
        # - R_world_cam: rotation matrix whose columns are world axes in camera frame
        # - t_world_cam: world origin position in camera frame

        # For a point P_world in world coordinates, its camera coordinates:
        # P_cam = R_world_cam @ P_world + t_world_cam  (if R_world_cam rotates world->cam)
        # But actually from the code, R_world_cam columns are world axes in camera frame
        # So R_world_cam @ P_world gives P_cam (without translation consideration for rotation)

        # Cube position in camera frame
        t_cube_cam = R_world_cam @ t_cube_world + t_world_cam

        # Cube axes in camera frame
        # R_cube_world columns are cube axes in world frame
        # R_world_cam @ (cube axis in world) = cube axis in camera
        R_cube_cam = R_world_cam @ R_cube_world

        # Draw axes at cube position
        origin = t_cube_cam.reshape(3, 1)
        x_end = origin + axis_length * R_cube_cam[:, 0:1]
        y_end = origin + axis_length * R_cube_cam[:, 1:2]
        z_end = origin + axis_length * R_cube_cam[:, 2:3]

        # Project to 2D
        origin_2d, _ = cv2.projectPoints(origin.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)
        x_2d, _ = cv2.projectPoints(x_end.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)
        y_2d, _ = cv2.projectPoints(y_end.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)
        z_2d, _ = cv2.projectPoints(z_end.T, np.zeros(3), np.zeros(3), K, DIST_COEFFS)

        origin_pt = tuple(origin_2d[0, 0].astype(int))
        x_pt = tuple(x_2d[0, 0].astype(int))
        y_pt = tuple(y_2d[0, 0].astype(int))
        z_pt = tuple(z_2d[0, 0].astype(int))

        # Draw with lighter colors to distinguish from world axes
        cv2.arrowedLine(img, origin_pt, x_pt, (100, 100, 255), line_width, tipLength=0.3)   # X - Light Red
        cv2.arrowedLine(img, origin_pt, y_pt, (100, 255, 100), line_width, tipLength=0.3)   # Y - Light Green
        cv2.arrowedLine(img, origin_pt, z_pt, (255, 100, 100), line_width, tipLength=0.3)   # Z - Light Blue

    def run(self):
        """Main detection loop."""
        print("\nCube World Observer running...")
        print("  Press 'q' to quit, 'r' to reset filters, 'w' to resample world frame\n")

        # Start world frame sampling on startup
        self.start_world_sampling()

        while True:
            _tA = time.perf_counter()
            ret = self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 100)
            if ret != 0:
                continue
            grab_dt = time.perf_counter() - _tA

            # --- Buffer backlog detection & recovery ---
            if grab_dt > BACKLOG_LATENCY_S:
                self._grab_slow_count += 1
                if self._grab_slow_count >= BACKLOG_COUNT:
                    # Flush: free current frame, then drain queued frames (bounded)
                    self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
                    flushed = 0
                    while flushed < BACKLOG_MAX_FLUSH:
                        r = self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 1)
                        if r != 0:
                            break
                        self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
                        flushed += 1
                    # Re-grab a fresh frame
                    ret = self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 100)
                    if ret != 0:
                        self._grab_slow_count = 0
                        continue
                    print(f"[FLUSH] buffer backlog detected (grab={grab_dt*1000:.1f}ms), "
                          f"drained {flushed} stale frames")
                    self._grab_slow_count = 0
            else:
                self._grab_slow_count = 0

            self.frame_count += 1
            nH = self.stOutFrame.stFrameInfo.nHeight
            nW = self.stOutFrame.stFrameInfo.nWidth
            data = string_at(self.stOutFrame.pBufAddr, self.stOutFrame.stFrameInfo.nFrameLen)
            bayer = np.frombuffer(data, dtype=np.uint8).reshape(nH, nW)

            # Demosaic to BGR (always, used both for min-channel gray and visualization)
            bgr = cv2.cvtColor(bayer, cv2.COLOR_BayerGB2BGR)

            # Min-channel: white→255, any color→≈0; robust for dark ArUco on white cube
            gray_min = np.minimum(np.minimum(bgr[:, :, 0], bgr[:, :, 1]), bgr[:, :, 2])

            # Optional CLAHE contrast enhancement (full-frame, before any ROI crop)
            if self._clahe is not None:
                gray = self._clahe.apply(gray_min)
            else:
                gray = gray_min

            color = bgr if self.visualize else None

            # Detect world AprilTag (skip when world frame is already fixed)
            if not self._world_fixed:
                R_world, t_world, world_corners = self.detect_world_tag(gray)
                world_detected = R_world is not None
                # Sampling mode: collect samples
                if world_detected:
                    self._world_samples_R.append(R_world)
                    self._world_samples_t.append(t_world)
                    if len(self._world_samples_R) >= self._world_sample_target:
                        self._finalize_world_frame()
            else:
                world_detected = True
                world_corners = None

            # Detect cube ArUco tags
            # In preview mode with world fixed, use software ROI crop for speed
            fast_roi = _cam_cfg.get('fast_roi')
            _use_sw_roi = (self.visualize and self._world_fixed
                           and fast_roi is not None)
            if _use_sw_roi:
                rx, ry = fast_roi['offset_x'], fast_roi['offset_y']
                rw, rh = fast_roi['width'], fast_roi['height']
                gray_roi = gray[ry:ry+rh, rx:rx+rw]
            else:
                gray_roi = gray
                rx, ry = 0, 0

            if self.aruco_detector is None:
                corners, ids, _ = cv2.aruco.detectMarkers(gray_roi, self.aruco_dict, parameters=self.aruco_params)
            else:
                corners, ids, _ = self.aruco_detector.detectMarkers(gray_roi)

            # Map corners back to full-frame coordinates
            if _use_sw_roi and ids is not None and len(ids) > 0:
                corners = [c + np.array([[[rx, ry]]], dtype=c.dtype) for c in corners]
            cube_quat_world = None
            cube_pos_world = None
            n_tags = 0

            if ids is not None and len(ids) > 0:
                mask = (ids.flatten() >= 0) & (ids.flatten() <= 23)
                if mask.any():
                    corners = [corners[i] for i in range(len(corners)) if mask[i]]
                    ids = ids[mask]
                else:
                    corners, ids = [], None

            # Corner-level EMA filter (state cache + reset hook before PnP)
            if ids is not None and len(ids) > 0:
                corners, ids = self.corner_filter.update(corners, ids)
            else:
                self.corner_filter.update([], None)

            R_cube_cam = None
            if ids is not None and len(ids) > 0:
                R_cube_cam, t_cube_cam, n_tags = self.detect_cube_pose(corners, ids)
                # _lost_frames managed inside detect_cube_pose on the success / fail branches
            else:
                # No cube markers this frame — still counts as a lost frame so the
                # IPPE disambiguation reset logic sees a fresh reacquire next time.
                self._lost_frames += 1

            if R_cube_cam is not None and self.world_pose is not None:
                R_cube_world, t_cube_world = self.transform_to_world_frame(R_cube_cam, t_cube_cam)
                if R_cube_world is not None:
                    # Apply cube frame correction if specified
                    # This corrects the difference between ArUco board axes and MuJoCo mesh axes
                    if CUBE_FRAME_CORRECTION is not None:
                        if isinstance(CUBE_FRAME_CORRECTION, str):
                            cube_correction_R = parse_axis_remap(CUBE_FRAME_CORRECTION)
                        else:
                            cube_correction_R = np.array(CUBE_FRAME_CORRECTION)
                        # R_cube_world_corrected = R_cube_world @ cube_correction_R.T
                        R_cube_world = R_cube_world @ cube_correction_R.T

                    # Frame: wrist-tag frame (observer-native). Consumers
                    # (RealHandEnv via CubeReceiver) treat this as tag-frame
                    # cube pose, fed directly to policy obs via the deploy-side
                    # override funcs in lib/real_hand_obs.py.
                    # Store for visualization
                    self._R_cube_world = R_cube_world
                    self._t_cube_world = t_cube_world

                    # R_cube_world: transforms from cube frame TO world frame
                    # For MuJoCo mocap body, we need the quaternion that represents
                    # cube orientation in world frame (rotation from cube to world)
                    rot = Rotation.from_matrix(R_cube_world)
                    quat = rot.as_quat()  # (x, y, z, w)

                    # Quaternion sign continuity: q and -q represent same rotation
                    # Choose sign to minimize distance from previous quaternion
                    if self.prev_quat is not None:
                        if np.dot(quat, self.prev_quat) < 0:
                            quat = -quat
                    self.prev_quat = quat.copy()

                    cube_quat_world = quat
                    cube_pos_world = t_cube_world

            # Publish via ZMQ.
            # Frame: wrist-tag frame (observer-native, from transform_to_world_frame).
            # Consumers (RealHandEnv via CubeReceiver) treat this as tag-frame
            # cube pose, fed directly to policy obs via the deploy-side override
            # funcs in lib/real_hand_obs.py — no mjworld round-trip.
            # Quaternion order: (x, y, z, w) — scipy convention. CubeReceiver
            # converts to MuJoCo (w, x, y, z) on receive (see zmq_bridge.py).
            if cube_quat_world is not None:
                q_xyzw = cube_quat_world  # scipy xyzw, as wuji computed it
                quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
                self.publisher.publish(cube_pos_world, quat_wxyz, world_fixed=self._world_fixed, cube_size=float(self._cube_size))

            # Visualization
            if self.visualize and color is not None:
                # Draw world frame axes (RGB = XYZ)
                if self._world_fixed and self.world_pose is not None:
                    self._draw_world_axes(color)

                if world_corners is not None:
                    cv2.polylines(color, [world_corners.astype(int)], True, (255, 0, 255), 3)

                # Draw detection ROI when active
                if _use_sw_roi:
                    cv2.rectangle(color, (rx, ry), (rx + rw, ry + rh), (0, 200, 200), 2)

                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(color, corners, ids)

                if n_tags > 0 and self._R_cube_world is not None:
                    # Draw cube axes in world frame (lighter colors)
                    self._draw_cube_axes_in_world(color, self._R_cube_world, self._t_cube_world)

                # World frame status display
                if not self._world_fixed:
                    # Sampling mode: show progress bar
                    n_samples = len(self._world_samples_R)
                    progress = n_samples / self._world_sample_target
                    bar_width = 200
                    bar_height = 20
                    cv2.rectangle(color, (10, 10), (10 + bar_width, 10 + bar_height), (50, 50, 50), -1)
                    cv2.rectangle(color, (10, 10), (10 + int(bar_width * progress), 10 + bar_height), (0, 255, 255), -1)
                    cv2.rectangle(color, (10, 10), (10 + bar_width, 10 + bar_height), (255, 255, 255), 1)
                    cv2.putText(color, f"World Sampling: {n_samples}/{self._world_sample_target}",
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # Fixed mode: show status
                    cv2.putText(color, "WORLD FIXED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    if world_detected:
                        cv2.putText(color, "(tag visible)", (180, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

                cv2.putText(color, f"Tags: {n_tags}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Display dominant face with color
                if self._dominant_face and self._dominant_face in FACE_COLORS:
                    face_name = self._dominant_face
                    color_name, face_bgr = FACE_COLORS[face_name]
                    # Draw color block + text
                    cv2.rectangle(color, (10, 75), (40, 105), face_bgr, -1)
                    cv2.rectangle(color, (10, 75), (40, 105), (255, 255, 255), 1)
                    cv2.putText(color, f"{face_name} ({color_name})", (50, 98),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_bgr, 2)

                if cube_quat_world is not None:
                    rpy = Rotation.from_quat(cube_quat_world).as_euler('xyz', degrees=True)
                    cv2.putText(color, f"RPY: ({rpy[0]:+.1f}, {rpy[1]:+.1f}, {rpy[2]:+.1f})", (10, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # FPS overlay (top-right)
                fps_color = (0, 255, 0) if self._display_fps >= 20 else (0, 165, 255)
                fps_text = f"FPS: {self._display_fps:.1f}"
                (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(color, fps_text, (color.shape[1] - tw - 10, th + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)

                # Key hints
                cv2.putText(color, "q:quit  r:reset  w:resample world  s:select ROI", (10, 755),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                cv2.imshow('Cube World Observer', cv2.resize(color, (960, 768)))

            self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)

            # Print status periodically
            now = time.time()
            if now - self.last_print_time > 2.0:
                elapsed = now - self.last_print_time if self.last_print_time > 0 else 1.0
                fps = (self.frame_count - self.last_frame_count) / elapsed
                self._display_fps = fps
                self.last_frame_count = self.frame_count
                if not self._world_fixed:
                    n_samples = len(self._world_samples_R)
                    print(f"[{self.frame_count:6d}] FPS: {fps:5.1f} | World Sampling: {n_samples}/{self._world_sample_target}")
                elif cube_quat_world is not None:
                    rpy = Rotation.from_quat(cube_quat_world).as_euler('xyz', degrees=True)
                    # cube_quat_world is (x,y,z,w) from scipy
                    qx, qy, qz, qw = cube_quat_world
                    px, py, pz = cube_pos_world
                    print(f"[{self.frame_count:6d}] FPS: {fps:5.1f} | World: FIXED | Tags: {n_tags} | "
                          f"Pos: ({px:+.4f}, {py:+.4f}, {pz:+.4f}) | "
                          f"Quat(wxyz): ({qw:+.4f}, {qx:+.4f}, {qy:+.4f}, {qz:+.4f}) | "
                          f"Quat(xyzw): ({qx:+.4f}, {qy:+.4f}, {qz:+.4f}, {qw:+.4f}) | "
                          f"RPY: ({rpy[0]:+6.1f}, {rpy[1]:+6.1f}, {rpy[2]:+6.1f})")
                else:
                    print(f"[{self.frame_count:6d}] FPS: {fps:5.1f} | World: FIXED | Cube: NOT DETECTED")
                self.last_print_time = now

            if self.visualize:
                key = cv2.pollKey() & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset cube filters only (not world frame)
                    self.filter_R.reset()
                    self.filter_t.reset()
                    self.prev_quat = None
                    print("Cube filters reset!")
                elif key == ord('w'):
                    # Resample world frame
                    self.start_world_sampling()
                    # Also reset cube filters since world frame changed
                    self.filter_R.reset()
                    self.filter_t.reset()
                    self.prev_quat = None
                elif key == ord('s'):
                    self._select_and_save_fast_roi(bgr)

    def _select_and_save_fast_roi(self, current_frame):
        """Open selectROI dialog, save selection to config/camera.yaml and apply live.

        Draws a rectangle on the full-resolution frame. Coordinates are in full
        sensor coordinates (offset_x/offset_y from 0,0). Persists to camera.yaml.
        """
        import os

        if not isinstance(current_frame, np.ndarray) or current_frame.ndim != 3:
            print(f"[ROI] ERROR: expected BGR image, got {type(current_frame).__name__}")
            return

        print("\n[ROI] Drag a rectangle on the frame. ENTER/SPACE to confirm, C to cancel.")
        # Use a resized preview for selection (same size as main imshow)
        display_size = (960, 768)
        display = cv2.resize(current_frame, display_size)
        scale_x = current_frame.shape[1] / display_size[0]
        scale_y = current_frame.shape[0] / display_size[1]

        x, y, w, h = cv2.selectROI(
            "Select ROI (ENTER/SPACE=confirm, C=cancel)", display,
            showCrosshair=True, fromCenter=False,
        )
        cv2.destroyWindow("Select ROI (ENTER/SPACE=confirm, C=cancel)")

        if w == 0 or h == 0:
            print("[ROI] Selection cancelled.")
            return

        # Scale back to full resolution
        offset_x = int(round(x * scale_x))
        offset_y = int(round(y * scale_y))
        width = int(round(w * scale_x))
        height = int(round(h * scale_y))

        # Some Hikvision cameras require width/height to be multiples of 4 or 8
        width = (width // 8) * 8
        height = (height // 8) * 8
        offset_x = (offset_x // 8) * 8
        offset_y = (offset_y // 8) * 8
        if width < 64 or height < 64:
            print(f"[ROI] Selection too small ({width}x{height}), ignored.")
            return

        print(f"[ROI] New fast_roi: offset=({offset_x},{offset_y}) size={width}x{height}")

        # Persist to config/camera.yaml via yaml load → modify → atomic write
        yaml_path = os.path.join(ROOT_DIR, "config", "camera.yaml")
        try:
            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f)
            cfg["fast_roi"] = {
                "offset_x": offset_x,
                "offset_y": offset_y,
                "width": width,
                "height": height,
            }
            tmp_path = yaml_path + ".tmp"
            with open(tmp_path, "w") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
            os.replace(tmp_path, yaml_path)
            print(f"[ROI] Saved to {yaml_path}")
        except Exception as exc:
            print(f"[ROI] Failed to save: {exc}")
            # Clean up stray tmp file if any
            try:
                os.remove(yaml_path + ".tmp")
            except OSError:
                pass
            return

        # Apply live: update in-memory config so software ROI uses new values
        _cam_cfg["fast_roi"] = {
            "offset_x": offset_x,
            "offset_y": offset_y,
            "width": width,
            "height": height,
        }
        print("[ROI] Applied. Next frames will use the new fast_roi.")

    def cleanup(self):
        """Release resources."""
        if self.visualize:
            cv2.destroyAllWindows()
        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        self.publisher.close()
        print("Cleanup done.")


def main():
    import argparse

    # Load config from file first
    cfg = load_observer_config()

    parser = argparse.ArgumentParser(description="Cube World Observer")
    parser.add_argument('--preview', action='store_true', help="Show preview window")
    parser.add_argument('--port', type=int, default=None, help="ZMQ port (override config)")
    parser.add_argument('--process-noise', type=float, default=None, help="SO3 Kalman Q (override config)")
    parser.add_argument('--measurement-noise', type=float, default=None, help="SO3 Kalman R (override config)")
    parser.add_argument('--alpha', type=float, default=None, help="Position LP alpha (override config)")
    parser.add_argument('--world-samples', type=int, default=WORLD_SAMPLE_FRAMES,
                        help=f"Number of frames to sample for world frame (default: {WORLD_SAMPLE_FRAMES})")
    parser.add_argument('--cube', type=str, default=None,
                        help="Cube tags config: a size suffix (e.g. '36', '40_5') "
                             "resolving to config/cube_tags<suffix>.json, or a literal "
                             "path. Default: config/cube_tags.json (54mm).")
    args = parser.parse_args()
    cube_config_path = resolve_cube_config_path(args.cube)

    # Use config values, CLI args override
    # ZMQ port from unified config_loader (control.yaml)
    port = args.port if args.port is not None else DEFAULT_CUBE_PORT
    process_noise = args.process_noise if args.process_noise is not None else cfg['rotation_filter']['process_noise']
    measurement_noise = args.measurement_noise if args.measurement_noise is not None else cfg['rotation_filter']['measurement_noise']
    alpha = args.alpha if args.alpha is not None else cfg['position_filter']['alpha']

    print(f"Filter params: process_noise={process_noise}, measurement_noise={measurement_noise}, alpha={alpha}")
    print(f"World frame sampling: {args.world_samples} frames")

    observer = CubeWorldObserver(
        visualize=args.preview,
        zmq_port=port,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        alpha=alpha,
        world_sample_frames=args.world_samples,
        cube_config_path=cube_config_path,
    )
    try:
        observer.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        observer.cleanup()


if __name__ == "__main__":
    main()
