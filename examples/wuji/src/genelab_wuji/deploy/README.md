# Wuji-hand deploy (Genesis-native)

A Genesis-native port of the `wuji-mjlab/deploy/reorient` pipeline. Two deliverables:

1. **real2sim** — reproduce the real cube's pose inside the Genesis sim.
2. **policy deploy** — run an exported ONNX policy to control the (real or mock) hand.

The pieces are decoupled via ZMQ (localhost):

```
 cube_world_observer ──cube pose (5555)──▶ play_real        (controls the hand)
   (camera, hardware)        │             toreal_viewer    (mirrors cube in sim)
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
| `scripts/play_real.py` | deploy control loop (real or mock hand) | glue |
| `scripts/toreal_viewer.py` | real2sim Genesis viewer | glue |
| `scripts/cube_world_observer.py` | camera → ZMQ vision pipeline | glue (hardware) |

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
uv pip install -e 'examples/wuji[deploy]'           # core (real2sim + control)
uv pip install -e 'examples/wuji[deploy,deploy-vision]'  # + camera observer
```

## Run

```bash
# 0) export a trained policy to ONNX
genelab export Genelab-Reorient-Wuji-Hand-v0 PATH/model.pt --format onnx --output policy.onnx

# 1) smoke-test the control loop, no hardware, no ZMQ
python -m genelab_wuji.deploy.scripts.play_real --ckpt policy.onnx --mock --no-zmq --steps 100

# 2) real2sim: mirror the real cube in the Genesis sim (needs GPU + display)
python -m genelab_wuji.deploy.scripts.cube_world_observer --camera 0   # terminal A
python -m genelab_wuji.deploy.scripts.toreal_viewer                    # terminal B

# 3) drive the real hand from the live observer feed
python -m genelab_wuji.deploy.scripts.play_real --ckpt policy.onnx --real
```

The vision observer here is a simplified port (single-marker PnP, no Kalman / dominant-face
fusion); for the production Hikvision rig, swap `_open_camera` and keep the rest.
