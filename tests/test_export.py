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


def test_export_task_forces_headless_vis(tmp_path: Path, monkeypatch: Any) -> None:
    """Regression: ``genelab export`` must force ``vis=False``.

    Trainable tasks set ``simulation.vis=play`` so their play_env opens the
    viewer; export never renders, so building that env on a display-less CI /
    server otherwise raises ``GenesisException: No display detected``. Drive the
    real ``export_task`` with the env build monkeypatched out (no Genesis runtime)
    and assert (a) the cfg handed to ``build_env`` has the viewer off and (b) the
    artifact + metadata are produced.
    """
    from genelab.cli import _export
    from genelab.rl import backends
    from genelab.rl.backends.base import InferenceSetup

    class _Sim:
        num_envs = 1
        vis = True  # as a trainable task's play_env configures it

    class _EnvCfg:
        def __init__(self) -> None:
            self.simulation = _Sim()

    env_cfg = _EnvCfg()

    class _ActionManager:
        total_action_dim = 3

    class _Env(_FakeEnv):
        def __init__(self, terms: dict[str, _FakeTermCfg]) -> None:
            super().__init__(terms)
            self.action_manager = _ActionManager()

    actor = _linear_actor(in_dim=4, out_dim=3)
    terms = {"a": _FakeTermCfg(dim=2, scale=1.0), "b": _FakeTermCfg(dim=2, scale=1.0)}

    captured: dict[str, Any] = {}

    def _build_env(cfg: Any) -> _Env:
        captured["vis"] = cfg.simulation.vis
        return _Env(terms)

    class _Backend:
        name = "fake"

        def make_inference_setup(self, _ctx: Any) -> InferenceSetup:
            return InferenceSetup(
                wrapper=None,
                adapter=None,
                policy=lambda o: o,
                actor_module=actor,
                actor_input_dim=4,
            )

    class _Task:
        class cfg:
            agent = object()

    class _Registry:
        def get(self, _task_id: str) -> _Task:
            return _Task()

    monkeypatch.setattr(_export, "ensure_project_cache", lambda: None)
    monkeypatch.setattr(_export, "resolve_env_cfg", lambda task_id, play: env_cfg)
    monkeypatch.setattr(_export, "build_env", _build_env)
    monkeypatch.setattr(_export, "TASKS", _Registry())
    monkeypatch.setattr(backends, "select_backend", lambda _agent_cfg: _Backend())

    out = tmp_path / "policy.ts"
    written = _export.export_task(
        "Fake-Task-v0", Path("/tmp/fake.pt"), format="torchscript", output=out
    )

    assert captured["vis"] is False
    assert env_cfg.simulation.vis is False
    assert written == out
    assert out.exists()
    assert out.with_suffix(out.suffix + ".metadata.json").exists()


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


def _fake_runner(alg: Any) -> Any:
    class _Runner:
        pass

    runner = _Runner()
    runner.alg = alg
    return runner


def test_extract_rsl_rl_actor_new_layout(tmp_path: Path) -> None:
    """Current rsl_rl exposes the actor on ``alg._raw_actor`` as an ``MLPModel``.

    Regression for the ``backend 'rsl_rl' did not expose an actor module`` failure:
    the extractor must source the actor from ``alg`` directly (not the removed
    ``alg.actor_critic``) and return a flat ``forward(obs) -> action`` module the
    exporter can serialize.
    """
    pytest.importorskip("rsl_rl")
    from tensordict import TensorDict

    from genelab.rl.backends.rsl_rl import _extract_rsl_rl_actor
    from genelab.rl.exporter import ExportConfig, export_policy
    from rsl_rl.models.mlp_model import MLPModel

    obs = TensorDict({"policy": torch.zeros(1, 4)}, batch_size=[1])
    actor = MLPModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        output_dim=3,
        hidden_dims=(8,),
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution"},
    )

    class _Alg:
        pass

    alg = _Alg()
    alg._raw_actor = actor

    module, in_dim = _extract_rsl_rl_actor(_fake_runner(alg), num_obs=99)
    assert in_dim == 4  # derived from the model, not the num_obs fallback
    # The returned module takes a flat obs tensor (not a TensorDict) and is
    # deterministic — the contract the exporter bakes normalization around.
    actions = module(torch.zeros(2, 4))
    assert actions.shape == (2, 3)

    terms = {"a": _FakeTermCfg(dim=2, scale=1.0), "b": _FakeTermCfg(dim=2, scale=1.0)}
    out = tmp_path / "policy.ts"
    export_policy(
        task_id="Rsl-Task-v0",
        env=_FakeEnv(terms),
        checkpoint=Path("/tmp/fake.pt"),
        actor=module,
        actor_input_dim=in_dim,
        action_dim=3,
        policy_group="policy",
        cfg=ExportConfig(format="torchscript", output=out),
    )
    assert out.exists()
    assert out.with_suffix(out.suffix + ".metadata.json").exists()
    loaded = torch.jit.load(str(out))
    assert loaded(torch.zeros(5, 4)).shape == (5, 3)


def test_extract_rsl_rl_actor_legacy_layout() -> None:
    """Older rsl_rl kept the actor under ``alg.actor_critic.actor`` — still supported."""
    actor = _linear_actor(in_dim=6, out_dim=2)

    class _ActorCritic:
        def __init__(self, actor: Any) -> None:
            self.actor = actor

    class _Alg:
        def __init__(self, ac: Any) -> None:
            self.actor_critic = ac

    from genelab.rl.backends.rsl_rl import _extract_rsl_rl_actor

    module, in_dim = _extract_rsl_rl_actor(_fake_runner(_Alg(_ActorCritic(actor))), num_obs=99)
    assert module is actor
    assert in_dim == 6  # first Linear's in_features, not the fallback


def test_extract_rsl_rl_actor_missing_returns_none() -> None:
    """No actor anywhere -> ``(None, num_obs)`` so the caller can raise a clear error."""
    from genelab.rl.backends.rsl_rl import _extract_rsl_rl_actor

    class _Alg:
        pass

    module, in_dim = _extract_rsl_rl_actor(_fake_runner(_Alg()), num_obs=7)
    assert module is None
    assert in_dim == 7
