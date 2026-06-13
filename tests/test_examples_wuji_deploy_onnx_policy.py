"""ONNX policy wrapper for deploy: load a GeneLab-exported policy and run it.

GeneLab's exporter writes an ONNX with input ``obs`` / output ``actions`` (batch
axis dynamic) plus a sibling ``<file>.metadata.json`` recording ``obs_dim`` and
``action_dim`` with normalization baked in. These tests build a tiny real ONNX so
the load / dim-introspection / single-forward path is exercised end to end.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")

from genelab_wuji.deploy.onnx_policy import ONNXPolicy  # noqa: E402

_OBS_DIM = 207
_ACTION_DIM = 20


def _export_tiny_policy(tmp_path: Path, obs_dim: int, action_dim: int) -> Path:
    """Export a 1-layer Linear policy to ONNX + sibling metadata.json (GeneLab fmt)."""
    model = torch.nn.Linear(obs_dim, action_dim)
    model.eval()
    onnx_path = tmp_path / "policy.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, obs_dim),
        str(onnx_path),
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
    )
    meta = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "action_range": [-1.0, 1.0],
        "normalization_baked": True,
    }
    (tmp_path / "policy.onnx.metadata.json").write_text(json.dumps(meta))
    return onnx_path


def test_loads_and_reports_dims(tmp_path: Path) -> None:
    onnx_path = _export_tiny_policy(tmp_path, _OBS_DIM, _ACTION_DIM)
    policy = ONNXPolicy(onnx_path)
    assert policy.input_dim == _OBS_DIM
    assert policy.action_dim == _ACTION_DIM


def test_single_forward_returns_action_vector(tmp_path: Path) -> None:
    onnx_path = _export_tiny_policy(tmp_path, _OBS_DIM, _ACTION_DIM)
    policy = ONNXPolicy(onnx_path)
    action = policy(np.zeros(_OBS_DIM, dtype=np.float32))
    assert action.shape == (_ACTION_DIM,)


def test_wrong_obs_dim_raises(tmp_path: Path) -> None:
    onnx_path = _export_tiny_policy(tmp_path, _OBS_DIM, _ACTION_DIM)
    policy = ONNXPolicy(onnx_path)
    with pytest.raises(ValueError, match="expected"):
        policy(np.zeros(_OBS_DIM - 1, dtype=np.float32))


def test_reads_metadata_sidecar(tmp_path: Path) -> None:
    onnx_path = _export_tiny_policy(tmp_path, _OBS_DIM, _ACTION_DIM)
    policy = ONNXPolicy(onnx_path)
    assert policy.metadata["obs_dim"] == _OBS_DIM
    assert policy.metadata["normalization_baked"] is True
