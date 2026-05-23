"""Unit tests for :mod:`genelab.rl.exporter`.

Cover the normalization-baking and serialization paths without a Genesis runtime:
build a fake actor + fake env that exposes the same ``observation_manager`` API
the exporter probes, then trace it to TorchScript / ONNX and verify load+forward
in the same process. Full end-to-end CLI tests (real env + checkpoint) live in
the integration verification block.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")


class _FakeTermCfg:
    """Mimics ``ObservationTermCfg``: scale/clip + a ``func`` that returns a tensor."""

    def __init__(
        self, dim: int, scale: float | None = None, clip: tuple[float, float] | None = None
    ) -> None:
        self.scale = scale
        self.clip = clip
        self.params: dict[str, Any] = {}
        self._dim = dim

    def func(self, env: Any) -> torch.Tensor:  # noqa: ARG002 - signature parity
        return torch.zeros(1, self._dim)


class _FakeObsManager:
    """Mimics ``ObservationManager``: ``cfg`` dict + ``compute()`` returning groups."""

    def __init__(self, group_name: str, terms: dict[str, _FakeTermCfg]) -> None:
        class _Group:
            def __init__(self, terms: dict[str, _FakeTermCfg]) -> None:
                self.terms = terms

        self.cfg = {group_name: _Group(terms)}
        self._group_name = group_name
        self._dim = sum(t._dim for t in terms.values())

    def compute(self) -> dict[str, torch.Tensor]:
        return {self._group_name: torch.zeros(1, self._dim)}


class _FakeEnv:
    def __init__(self, terms: dict[str, _FakeTermCfg]) -> None:
        self.observation_manager = _FakeObsManager("policy", terms)

    def close(self) -> None: ...


def _linear_actor(in_dim: int, out_dim: int) -> torch.nn.Module:
    """A 2-layer MLP with deterministic, easily-invertible weights for shape checks."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, out_dim),
    )


def test_export_torchscript_smoke(tmp_path: Path) -> None:
    from genelab.rl.exporter import ExportConfig, export_policy

    terms = {
        "joint_pos": _FakeTermCfg(dim=2, scale=1.0, clip=None),
        "joint_vel": _FakeTermCfg(dim=2, scale=0.5, clip=(-2.0, 2.0)),
    }
    env = _FakeEnv(terms)
    actor = _linear_actor(in_dim=4, out_dim=3)

    out = tmp_path / "policy.ts"
    written = export_policy(
        task_id="Fake-Task-v0",
        env=env,
        checkpoint=Path("/tmp/fake.pt"),
        actor=actor,
        actor_input_dim=4,
        action_dim=3,
        policy_group="policy",
        cfg=ExportConfig(format="torchscript", output=out),
    )
    assert written == out
    assert out.exists()

    loaded = torch.jit.load(str(out))
    actions = loaded(torch.zeros(1, 4))
    assert actions.shape == (1, 3)
    actions_batch = loaded(torch.zeros(4, 4))
    assert actions_batch.shape == (4, 3)


def test_export_metadata_schema(tmp_path: Path) -> None:
    from genelab.rl.exporter import ExportConfig, export_policy

    terms = {
        "joint_pos": _FakeTermCfg(dim=2, scale=1.0, clip=None),
        "joint_vel": _FakeTermCfg(dim=2, scale=0.5, clip=(-2.0, 2.0)),
    }
    env = _FakeEnv(terms)
    actor = _linear_actor(in_dim=4, out_dim=3)

    out = tmp_path / "policy.ts"
    export_policy(
        task_id="Fake-Task-v0",
        env=env,
        checkpoint=Path("/tmp/fake.pt"),
        actor=actor,
        actor_input_dim=4,
        action_dim=3,
        policy_group="policy",
        cfg=ExportConfig(format="torchscript", output=out),
    )
    meta_path = out.with_suffix(out.suffix + ".metadata.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["task"] == "Fake-Task-v0"
    assert meta["action_dim"] == 3
    assert meta["normalization_baked"] is True
    assert meta["format"] == "torchscript"
    group = meta["obs_groups"]["policy"]
    assert group["dim"] == 4
    term_names = [t["name"] for t in group["terms"]]
    assert term_names == ["joint_pos", "joint_vel"]
    vel_term = next(t for t in group["terms"] if t["name"] == "joint_vel")
    assert vel_term["scale"] == 0.5
    assert vel_term["clip"] == [-2.0, 2.0]


def test_export_normalization_baked(tmp_path: Path) -> None:
    """A term with scale=0.5 should produce halved obs to the actor."""
    from genelab.rl.exporter import ExportConfig, export_policy

    terms = {"x": _FakeTermCfg(dim=3, scale=0.5, clip=None)}
    env = _FakeEnv(terms)

    # Actor that just returns the input as-is so we can read back the scaled obs.
    class _Identity(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    out = tmp_path / "policy.ts"
    export_policy(
        task_id="Fake-Task-v0",
        env=env,
        checkpoint=Path("/tmp/fake.pt"),
        actor=_Identity(),
        actor_input_dim=3,
        action_dim=3,
        policy_group="policy",
        cfg=ExportConfig(format="torchscript", output=out),
    )
    loaded = torch.jit.load(str(out))
    raw = torch.tensor([[2.0, 4.0, 6.0]])
    scaled = loaded(raw)
    # scale=0.5 baked in -> obs to actor = raw * 0.5
    assert torch.allclose(scaled, raw * 0.5)


def test_export_clip_baked(tmp_path: Path) -> None:
    """A term with clip=(-1,1) should clamp the obs fed to the actor."""
    from genelab.rl.exporter import ExportConfig, export_policy

    terms = {"x": _FakeTermCfg(dim=2, scale=None, clip=(-1.0, 1.0))}
    env = _FakeEnv(terms)

    class _Identity(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    out = tmp_path / "policy.ts"
    export_policy(
        task_id="Fake-Task-v0",
        env=env,
        checkpoint=Path("/tmp/fake.pt"),
        actor=_Identity(),
        actor_input_dim=2,
        action_dim=2,
        policy_group="policy",
        cfg=ExportConfig(format="torchscript", output=out),
    )
    loaded = torch.jit.load(str(out))
    raw = torch.tensor([[-5.0, 5.0]])
    clipped = loaded(raw)
    assert torch.allclose(clipped, torch.tensor([[-1.0, 1.0]]))


def test_export_no_backend_dependency_leak(tmp_path: Path) -> None:
    """Loading the exported TorchScript file in a clean subprocess must not import rsl_rl/skrl/sb3."""
    from genelab.rl.exporter import ExportConfig, export_policy

    terms = {"x": _FakeTermCfg(dim=4, scale=1.0, clip=None)}
    env = _FakeEnv(terms)
    actor = _linear_actor(in_dim=4, out_dim=2)
    out = tmp_path / "policy.ts"
    export_policy(
        task_id="Fake-Task-v0",
        env=env,
        checkpoint=Path("/tmp/fake.pt"),
        actor=actor,
        actor_input_dim=4,
        action_dim=2,
        policy_group="policy",
        cfg=ExportConfig(format="torchscript", output=out),
    )

    script = f"""
import sys
import torch
m = torch.jit.load({str(out)!r})
out = m(torch.zeros(1, 4))
assert out.shape == (1, 2), out.shape
banned = ("rsl_rl", "skrl", "stable_baselines3")
leaked = [n for n in banned if n in sys.modules]
assert not leaked, f"backend leaked into export: {{leaked}}"
print("OK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OK" in completed.stdout


def test_export_onnx_smoke(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    from genelab.rl.exporter import ExportConfig, export_policy

    terms = {"x": _FakeTermCfg(dim=4, scale=1.0, clip=None)}
    env = _FakeEnv(terms)
    actor = _linear_actor(in_dim=4, out_dim=2)
    out = tmp_path / "policy.onnx"
    export_policy(
        task_id="Fake-Task-v0",
        env=env,
        checkpoint=Path("/tmp/fake.pt"),
        actor=actor,
        actor_input_dim=4,
        action_dim=2,
        policy_group="policy",
        cfg=ExportConfig(format="onnx", output=out, opset=17),
    )
    import onnx

    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    meta_path = out.with_suffix(out.suffix + ".metadata.json")
    meta = json.loads(meta_path.read_text())
    assert meta["format"] == "onnx"
    assert meta["opset"] == 17
