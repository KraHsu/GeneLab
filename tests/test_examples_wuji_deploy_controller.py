"""End-to-end deploy control loop, headless (mock hand + self-driven ZMQ sources).

``DeployController`` ties the pieces together: read encoders -> build policy obs
(cube/goal from the observer feed) -> ONNX policy -> EMA action -> write target.
This is the "model deploy controls the hand" path; ``play_real.py`` wires the same
controller to the real hand, real ZMQ, and a Genesis viewer.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("zmq")
torch = pytest.importorskip("torch")

from genelab_wuji.deploy.config import default_joint_pos  # noqa: E402
from genelab_wuji.deploy.controller import DeployController  # noqa: E402
from genelab_wuji.deploy.hand_driver import MockHandDriver  # noqa: E402
from genelab_wuji.deploy.onnx_policy import ONNXPolicy  # noqa: E402
from genelab_wuji.deploy.zmq_bridge import CubeReceiver, GoalReceiver  # noqa: E402

_OBS_DIM = 207
_N = 20


def _export_zero_policy(tmp_path: Path) -> Path:
    """A policy that always outputs zeros (so post-warmup target == default)."""
    lin = torch.nn.Linear(_OBS_DIM, _N)
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.zero_()
    lin.eval()
    onnx_path = tmp_path / "policy.onnx"
    torch.onnx.export(
        lin,
        torch.zeros(1, _OBS_DIM),
        str(onnx_path),
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
    )
    (tmp_path / "policy.onnx.metadata.json").write_text(
        json.dumps({"obs_dim": _OBS_DIM, "action_dim": _N})
    )
    return onnx_path


def _make_controller(tmp_path: Path, **kw) -> tuple[DeployController, MockHandDriver]:
    policy = ONNXPolicy(_export_zero_policy(tmp_path))
    driver = MockHandDriver()
    cube = CubeReceiver(connect=False)
    goal = GoalReceiver(connect=False)
    ctrl = DeployController(
        policy=policy,
        driver=driver,
        cube_source=cube,
        goal_source=goal,
        default_joint_pos=default_joint_pos(),
        **kw,
    )
    return ctrl, driver


def test_step_returns_action_and_writes_target_to_hand(tmp_path: Path) -> None:
    ctrl, driver = _make_controller(tmp_path, warmup_steps=0)
    ctrl.reset()
    info = ctrl.step()
    assert info["action"].shape == (_N,)
    assert info["target"].shape == (_N,)
    # The hand received exactly the processed target.
    assert np.allclose(driver.read_encoders(), info["target"])


def test_zero_policy_post_warmup_drives_hand_to_default(tmp_path: Path) -> None:
    # A zero action -> target == default joint pos once warmup has elapsed.
    ctrl, driver = _make_controller(tmp_path, warmup_steps=0)
    ctrl.reset()
    for _ in range(5):
        ctrl.step()
    assert np.allclose(driver.read_encoders(), default_joint_pos(), atol=1e-5)
