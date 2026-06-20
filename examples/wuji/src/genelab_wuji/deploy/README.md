# Wuji-hand deploy (Genesis-native)

A Genesis-native port of the `wuji-mjlab/deploy/reorient` pipeline. Two deliverables:

1. **real2sim** — reproduce the real cube's pose inside the Genesis sim.
2. **policy deploy** — run an exported ONNX policy to control the (real or mock) hand.

The pieces are decoupled via ZMQ (localhost):

```
 cube_world_observer ──cube pose (5555)──▶ play_real        (controls the hand)
  (Hikvision camera)         │             toreal_viewer    (mirrors cube in sim)
                             └──────────────▶
 toreal_viewer ──goal (5556)──▶ play_real
```

## Architecture

| Module | Responsibility | Tested |
|---|---|---|
| `frame_transform.py` | wxyz quat math + `cube_cam_to_tag` (camera→wrist-tag lift) | ✅ |
| `real2sim.py` | `tag_pose_in_world`, `cube_pose_in_tag_to_world` (sim reproduction) | ✅ |
| `zmq_bridge.py` | cube/goal pub-sub + xyzw↔wxyz + last-valid cache | ✅ |
| `obs.py` | `DeployObsBuilder` (207-dim policy obs + 3-step history) | ✅ |
| `action.py` | `ActionProcessor` (offset + clamp + EMA + warmup) | ✅ |
| `onnx_policy.py` | `ONNXPolicy` (GeneLab metadata format) | ✅ |
| `hand_driver.py` | `HandDriverBase` / `MockHandDriver` / `WujiHandDriver` | ✅ (mock) |
| `controller.py` | `DeployController` (closed-loop step) | ✅ |
| `camera_config.py` | Hikvision intrinsics/ROI/capture from `config/camera.yaml` | glue (hardware) |
| `cube_geom.py` | cube_tags JSON resolution (`config/cube_tags.json`) | glue |
| `scripts/hand_utils.py` | `check` (read-only bridge test) / `home` (3s ramp to grasp pose) | glue (hardware) |
| `scripts/calib_check.py` | static calib viewer: live hand (encoders) + cube vs. digital twin | glue (hardware) |
| `scripts/play_real.py` | deploy control loop + goal modes + success monitor + Genesis mirror (real/mock) | glue |
| `scripts/toreal_viewer.py` | real2sim Genesis viewer | glue |
| `scripts/cube_world_observer.py` | Hikvision camera → ArUco board + SO3 Kalman → ZMQ cube pose | glue (hardware) |

The pure-software core is numpy-only and runs headlessly (no Genesis, no hardware),
so all frame/obs/action/policy logic is unit-tested in `tests/test_examples_wuji_deploy_*.py`.

### Key conventions

- **Quaternions**: wxyz everywhere internally; the cube wire format is scipy xyzw and
  is converted at the ZMQ boundary (`cube_pose_from_msg` / `cube_msg_from_pose`).
- **Tag frame**: the observer reports the cube already in the wrist-AprilTag frame —
  the exact frame the policy was trained on — so the deploy obs needs **no forward
  kinematics**.
- **6D goal error**: matches the GeneLab training encoding (`matrix_to_rotation_6d`,
  first two matrix rows), pinned against the real training math in the tests.
- **Joint order**: `finger1_joint1..4, finger2...` (= `wujihandpy` (5,4) row-major), so
  no remap between policy and hardware.

## Install

```bash
uv pip install -e 'examples/wuji[deploy]'                 # core (real2sim + control)
uv pip install -e 'examples/wuji[deploy,deploy-vision]'   # + camera observer
uv pip install -e 'examples/wuji[deploy,deploy-hand]'     # + real Wuji hand SDK (wujihandpy)
```

The cube observer also needs the **Hikvision MVS SDK** (system install, not pip — same
as wuji-mjlab). Install from <https://www.hikrobotics.com> (default `/opt/MVS`) and source
its environment before running the observer:

```bash
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib/32:$LD_LIBRARY_PATH
# (or: source /opt/MVS/bin/set_env_path.sh /opt/MVS)
# If MvImport lives elsewhere: export MVS_PYTHON_PATH=/path/to/dir/containing/MvImport
```

## Run

```bash
# 0) export a trained policy to ONNX
uv run genelab export Genelab-Reorient-Wuji-Hand-v0 PATH/model.pt --format onnx --out policy.onnx

# 1) smoke-test the control loop, no hardware, no ZMQ, no viewer
uv run python -m genelab_wuji.deploy.scripts.play_real --ckpt policy.onnx --mock --no-zmq --no-viewer --steps 100

# 1.5) bring up the real hand bridge (needs wujihandpy): check first, then home
uv run python -m genelab_wuji.deploy.scripts.hand_utils check   # READ-ONLY: connection + encoder sanity
uv run python -m genelab_wuji.deploy.scripts.hand_utils home    # 3s ease-in-out ramp to the grasp pose

# 2) vision: detect the cube and publish its tag-frame pose on ZMQ:5555 (needs MVS env)
uv run python -m genelab_wuji.deploy.scripts.cube_world_observer --preview   # terminal A
uv run python -m genelab_wuji.deploy.scripts.toreal_viewer                   # terminal B (real2sim mirror)

# 2.5) calibration check: home the hand, render live hand + observed cube in the twin
uv run python -m genelab_wuji.deploy.scripts.calib_check                     # (needs the observer running)

# 3) drive the real hand from the live observer feed (Genesis mirror viewer on by default,
#    showing the live hand + observed cube + goal; pass --no-viewer for headless).
#    goal modes: --goal-mode random (uniform-SO3, resampled on success) |
#                fixed --goal-quat w,x,y,z | external (goal from toreal_viewer ZMQ)
uv run python -m genelab_wuji.deploy.scripts.play_real --ckpt policy.onnx --real --goal-mode random
```

`play_real` mirrors the live hand (encoders) + observed cube + goal in a Genesis viewer
by default (`--no-viewer` to disable). It reuses the same kinematic, physics-free refresh
as `calib_check`, so the mirror just reflects reality. The control core itself is numpy-only
and runs headlessly under `--no-viewer`.

The cube observer is a faithful port of the production wuji-mjlab pipeline (Hikvision MVS
capture, multi-face ArUco board fusion, SO3 Kalman + position low-pass + corner EMA, world
auto-sampling, fast ROI, OpenCV preview). It publishes the cube pose in the wrist-tag frame
in the exact same ZMQ schema GeneLab's `CubeReceiver` consumes. Tuning lives in
`config/observer.yaml`; camera intrinsics/ROI in `config/camera.yaml`; cube tag layout in
`config/cube_tags.json`. For a non-Hikvision camera, swap the MVS capture in `run()`.
