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
    """Mimics ``ObservationManager``: ``cfg`` dict + ``compute()`` returning groups.

    Takes a mapping of group name -> term dict, so it serves both single-group
    (flat) and multi-group (goal-conditioned HER) export tests.
    """

    def __init__(
        self,
        groups: dict[str, dict[str, _FakeTermCfg]],
        history: dict[str, int] | None = None,
    ) -> None:
        history = history or {}

        class _Group:
            def __init__(self, terms: dict[str, _FakeTermCfg], history_length: int) -> None:
                self.terms = terms
                self.history_length = history_length

        self.cfg = {name: _Group(terms, history.get(name, 1)) for name, terms in groups.items()}
        self._dims = {
            name: sum(t._dim for t in terms.values()) * history.get(name, 1)
            for name, terms in groups.items()
        }

    def compute(self) -> dict[str, torch.Tensor]:
        return {name: torch.zeros(1, dim) for name, dim in self._dims.items()}


class _FakeEnv:
    def __init__(
        self,
        groups: dict[str, dict[str, _FakeTermCfg]],
        history: dict[str, int] | None = None,
    ) -> None:
        self.observation_manager = _FakeObsManager(groups, history)

    def close(self) -> None: ...


def _linear_actor(in_dim: int, out_dim: int) -> torch.nn.Module:
    """A 2-layer MLP with deterministic, easily-invertible weights for shape checks."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, out_dim),
    )


def test_resolve_obs_specs_accounts_for_history_length() -> None:
    """A history-stacked group exports an obs_dim of ``h * base`` with per-frame scale/clip.

    The exported policy must take the same frame-major stacked vector the env feeds the
    actor; otherwise the sim2sim deployment hands the network the wrong-width input.
    """
    from genelab.rl.exporter import _resolve_obs_specs, _scale_clip_vecs

    terms = {
        "a": _FakeTermCfg(dim=3, scale=2.0, clip=None),
        "b": _FakeTermCfg(dim=1, scale=None, clip=(-1.0, 1.0)),
    }
    env = _FakeEnv({"policy": terms}, history={"policy": 3})
    specs, obs_dim, group_specs = _resolve_obs_specs(env, ["policy"])
    # (3 + 1) base dims, stacked 3 frames.
    assert obs_dim == 12
    assert group_specs[0].dim == 12
    # Scale/clip repeat the per-frame block [a:2,2,2, b:1] across all 3 frames.
    scale, lo, hi = _scale_clip_vecs(specs, obs_dim)
    assert torch.allclose(scale, torch.tensor([2.0, 2.0, 2.0, 1.0]).repeat(3))
    # b's clip lands at the last column of each frame (indices 3, 7, 11).
    assert lo[3] == -1.0 and lo[7] == -1.0 and lo[11] == -1.0
    assert hi[3] == 1.0 and hi[7] == 1.0 and hi[11] == 1.0


def test_export_torchscript_smoke(tmp_path: Path) -> None:
    from genelab.rl.exporter import ExportConfig, export_policy

    terms = {
        "joint_pos": _FakeTermCfg(dim=2, scale=1.0, clip=None),
        "joint_vel": _FakeTermCfg(dim=2, scale=0.5, clip=(-2.0, 2.0)),
    }
    env = _FakeEnv({"policy": terms})
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
    env = _FakeEnv({"policy": terms})
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
    env = _FakeEnv({"policy": terms})

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
    env = _FakeEnv({"policy": terms})

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
    env = _FakeEnv({"policy": terms})
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
        def __init__(self, groups: dict[str, dict[str, _FakeTermCfg]]) -> None:
            super().__init__(groups)
            self.action_manager = _ActionManager()

    actor = _linear_actor(in_dim=4, out_dim=3)
    terms = {"a": _FakeTermCfg(dim=2, scale=1.0), "b": _FakeTermCfg(dim=2, scale=1.0)}

    captured: dict[str, Any] = {}

    def _build_env(cfg: Any) -> _Env:
        captured["vis"] = cfg.simulation.vis
        return _Env({"policy": terms})

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
    env = _FakeEnv({"policy": terms})
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

    module, in_dim, is_recurrent = _extract_rsl_rl_actor(_fake_runner(alg), num_obs=99)
    assert in_dim == 4  # derived from the model, not the num_obs fallback
    assert is_recurrent is False
    # The returned module takes a flat obs tensor (not a TensorDict) and is
    # deterministic — the contract the exporter bakes normalization around.
    actions = module(torch.zeros(2, 4))
    assert actions.shape == (2, 3)

    terms = {"a": _FakeTermCfg(dim=2, scale=1.0), "b": _FakeTermCfg(dim=2, scale=1.0)}
    out = tmp_path / "policy.ts"
    export_policy(
        task_id="Rsl-Task-v0",
        env=_FakeEnv({"policy": terms}),
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

    module, in_dim, is_recurrent = _extract_rsl_rl_actor(
        _fake_runner(_Alg(_ActorCritic(actor))), num_obs=99
    )
    assert module is actor
    assert in_dim == 6  # first Linear's in_features, not the fallback
    assert is_recurrent is False


def test_extract_rsl_rl_actor_missing_returns_none() -> None:
    """No actor anywhere -> ``(None, num_obs)`` so the caller can raise a clear error."""
    from genelab.rl.backends.rsl_rl import _extract_rsl_rl_actor

    class _Alg:
        pass

    module, in_dim, is_recurrent = _extract_rsl_rl_actor(_fake_runner(_Alg()), num_obs=7)
    assert module is None
    assert in_dim == 7
    assert is_recurrent is False


# -- rsl_rl recurrent (RNN/LSTM/GRU) export ---------------------------------------


def _rnn_actor(rnn_type: str, obs_dim: int, act_dim: int, hidden: int, layers: int) -> Any:
    """Build a real rsl_rl ``RNNModel`` (the object the exporter's recurrent path expects)."""
    from rsl_rl.models import RNNModel
    from tensordict import TensorDict

    obs = TensorDict({"policy": torch.zeros(4, obs_dim)}, batch_size=[4])
    model = RNNModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        act_dim,
        hidden_dims=(16, 16),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution"},
        rnn_type=rnn_type,
        rnn_hidden_dim=hidden,
        rnn_num_layers=layers,
    )
    model.eval()
    return model


def test_extract_rsl_rl_actor_flags_recurrent() -> None:
    """An ``RNNModel`` actor is returned raw with ``is_recurrent=True`` (not as_jit'd)."""
    pytest.importorskip("rsl_rl")
    from genelab.rl.backends.rsl_rl import _extract_rsl_rl_actor

    actor = _rnn_actor("lstm", obs_dim=4, act_dim=3, hidden=8, layers=1)

    class _Alg:
        pass

    alg = _Alg()
    alg._raw_actor = actor
    module, in_dim, is_recurrent = _extract_rsl_rl_actor(_fake_runner(alg), num_obs=99)
    assert is_recurrent is True
    assert in_dim == 4
    assert module is actor  # raw model, so the exporter can call as_jit()/as_onnx()


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_export_recurrent_torchscript(tmp_path: Path, rnn_type: str) -> None:
    """Recurrent TS export scripts the stateful wrapper: forward(obs)->actions + reset()."""
    pytest.importorskip("rsl_rl")
    from genelab.rl.exporter import ExportConfig, export_policy

    hidden, layers = 8, 2
    actor = _rnn_actor(rnn_type, obs_dim=4, act_dim=3, hidden=hidden, layers=layers)
    terms = {"a": _FakeTermCfg(dim=2, scale=2.0), "b": _FakeTermCfg(dim=2, clip=(-1.0, 1.0))}
    out = tmp_path / "policy.ts"
    export_policy(
        task_id="Rnn-Task-v0",
        env=_FakeEnv({"policy": terms}),
        checkpoint=Path("/tmp/fake.pt"),
        actor=actor,
        actor_input_dim=4,
        action_dim=3,
        policy_group="policy",
        cfg=ExportConfig(format="torchscript", output=out),
        is_recurrent=True,
    )
    loaded = torch.jit.load(str(out))
    loaded.eval()
    a1 = loaded(torch.zeros(1, 4))
    assert a1.shape == (1, 3)
    # Hidden state evolves across calls; reset() returns it to the post-first-step value.
    a2 = loaded(torch.ones(1, 4))
    loaded.reset()
    a3 = loaded(torch.ones(1, 4))
    assert not torch.allclose(a2, a3)

    meta = json.loads((out.with_suffix(out.suffix + ".metadata.json")).read_text())
    assert meta["is_recurrent"] is True
    assert meta["recurrent"]["rnn_type"] == rnn_type
    assert meta["recurrent"]["rnn_num_layers"] == layers
    assert meta["recurrent"]["rnn_hidden_dim"] == hidden


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_export_recurrent_onnx(tmp_path: Path, rnn_type: str) -> None:
    """Recurrent ONNX export exposes explicit hidden-state ports (h_in/c_in -> h_out/c_out)."""
    pytest.importorskip("rsl_rl")
    pytest.importorskip("onnx")
    import onnx

    from genelab.rl.exporter import ExportConfig, export_policy

    actor = _rnn_actor(rnn_type, obs_dim=4, act_dim=3, hidden=8, layers=1)
    terms = {"a": _FakeTermCfg(dim=2, scale=1.0), "b": _FakeTermCfg(dim=2, scale=1.0)}
    out = tmp_path / "policy.onnx"
    export_policy(
        task_id="Rnn-Task-v0",
        env=_FakeEnv({"policy": terms}),
        checkpoint=Path("/tmp/fake.pt"),
        actor=actor,
        actor_input_dim=4,
        action_dim=3,
        policy_group="policy",
        cfg=ExportConfig(format="onnx", output=out, opset=17),
        is_recurrent=True,
    )
    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    ins = [i.name for i in model.graph.input]
    outs = [o.name for o in model.graph.output]
    assert ins[0] == "obs" and outs[0] == "actions"
    assert ("c_in" in ins) == (rnn_type == "lstm")
    assert ("c_out" in outs) == (rnn_type == "lstm")

    meta = json.loads((out.with_suffix(out.suffix + ".metadata.json")).read_text())
    assert meta["recurrent"]["onnx_inputs"] == ins
    assert meta["recurrent"]["onnx_outputs"] == outs


def test_recurrent_train_to_export_end_to_end(tmp_path: Path) -> None:
    """The whole recurrent chain on the real rsl_rl runner, no Genesis:

    cfg (``rnn_type``) -> ``OnPolicyRunner`` builds an ``RNNModel`` -> train -> save ->
    reload -> ``_extract_rsl_rl_actor`` flags it recurrent -> ``export_policy`` writes a
    TorchScript file that loads, runs, and ``reset()``s. Drives the runner with a tiny
    fake VecEnv so it exercises the recurrent rollout-storage path without a simulator.
    """
    pytest.importorskip("rsl_rl")
    from tensordict import TensorDict

    from genelab.rl.backends.rsl_rl import _extract_rsl_rl_actor, _runner_cfg_to_dict
    from genelab.rl.config import (
        RslRlModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
    )
    from genelab.rl.exporter import ExportConfig, export_policy
    from rsl_rl.runners import OnPolicyRunner

    ne, obs_dim, act_dim = 8, 6, 2

    class _FakeVecEnv:
        """Minimal rsl_rl VecEnv with ``policy`` + ``critic`` obs groups and toy dynamics."""

        def __init__(self) -> None:
            self.num_envs, self.num_actions = ne, act_dim
            self.max_episode_length = 16
            self.episode_length_buf = torch.zeros(ne, dtype=torch.long)
            self.device = "cpu"
            self.cfg: dict[str, Any] = {}

        def _obs(self) -> Any:
            return TensorDict(
                {"policy": torch.randn(ne, obs_dim), "critic": torch.randn(ne, obs_dim)},
                batch_size=[ne],
            )

        def get_observations(self) -> Any:
            return self._obs()

        def reset(self) -> Any:
            return self._obs(), {"observations": {}}

        def step(self, _actions: Any) -> Any:
            self.episode_length_buf += 1
            dones = self.episode_length_buf >= self.max_episode_length
            self.episode_length_buf[dones] = 0
            extras = {"observations": {}, "time_outs": dones.clone()}
            return self._obs(), torch.randn(ne), dones.float(), extras

    def _rnn_cfg() -> Any:
        return RslRlModelCfg(
            rnn_type="lstm",
            rnn_hidden_dim=32,
            rnn_num_layers=1,
            hidden_dims=(32,),
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution"},
        )

    agent = RslRlOnPolicyRunnerCfg(actor=_rnn_cfg(), critic=_rnn_cfg(), num_steps_per_env=8)
    agent.algorithm = RslRlPpoAlgorithmCfg(num_learning_epochs=1, num_mini_batches=1)

    env = _FakeVecEnv()
    # A fresh cfg dict per runner: OnPolicyRunner pops keys (e.g. class_name) in place.
    runner = OnPolicyRunner(env, _runner_cfg_to_dict(agent), log_dir=str(tmp_path), device="cpu")
    assert type(runner.alg.actor).__name__ == "RNNModel"
    assert runner.alg.actor.is_recurrent is True
    runner.learn(num_learning_iterations=2)
    ckpt = tmp_path / "model.pt"
    runner.save(str(ckpt))

    runner2 = OnPolicyRunner(env, _runner_cfg_to_dict(agent), log_dir=None, device="cpu")
    runner2.load(str(ckpt))
    policy = runner2.get_inference_policy(device="cpu")
    policy.reset(torch.ones(ne, dtype=torch.bool))  # reset hook works on the inference policy
    module, in_dim, is_recurrent = _extract_rsl_rl_actor(runner2, num_obs=obs_dim)
    assert is_recurrent is True and in_dim == obs_dim

    terms = {"x": _FakeTermCfg(dim=obs_dim, scale=1.0)}
    out = tmp_path / "policy.ts"
    export_policy(
        task_id="Rnn-Trained-v0",
        env=_FakeEnv({"policy": terms}),
        checkpoint=ckpt,
        actor=module,
        actor_input_dim=in_dim,
        action_dim=act_dim,
        policy_group="policy",
        cfg=ExportConfig(format="torchscript", output=out),
        is_recurrent=is_recurrent,
    )
    loaded = torch.jit.load(str(out))
    loaded.eval()
    assert loaded(torch.zeros(1, obs_dim)).shape == (1, act_dim)
    loaded.reset()  # exported module exposes the episode-boundary reset


# -- SB3 goal-conditioned (SAC+HER) Dict-obs export -------------------------------


def _franka_style_env() -> _FakeEnv:
    """Fake env with the Franka SAC+HER groups: policy(5) / achieved_goal(2) / desired_goal(2)."""
    return _FakeEnv(
        {
            "policy": {
                "a": _FakeTermCfg(dim=2, scale=1.0),
                "b": _FakeTermCfg(dim=3, scale=0.5),
            },
            "achieved_goal": {"ag": _FakeTermCfg(dim=2)},
            "desired_goal": {"dg": _FakeTermCfg(dim=2)},
        }
    )


def _tiny_her_sac_model() -> Any:
    """A tiny SB3 SAC ``MultiInputPolicy`` on a Dict-obs env (no Genesis needed).

    Mirrors the Franka SAC+HER obs layout: ``observation``(5) / ``achieved_goal``(2)
    / ``desired_goal``(2). Imported function-locally so SB3 (and transitively cv2)
    stays out of pytest collection — see ``tests/test_sb3_pipeline.py``.
    """
    pytest.importorskip("stable_baselines3")
    import gymnasium
    import numpy as np
    from stable_baselines3 import SAC

    class _TinyGoalEnv(gymnasium.Env):
        def __init__(self) -> None:
            self.observation_space = gymnasium.spaces.Dict(
                {
                    "observation": gymnasium.spaces.Box(-1.0, 1.0, (5,), np.float32),
                    "achieved_goal": gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32),
                    "desired_goal": gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32),
                }
            )
            self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (3,), np.float32)

        def _obs(self) -> dict[str, Any]:
            return {k: self.observation_space[k].sample() * 0 for k in self.observation_space}

        def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, dict[str, Any]]:
            super().reset(seed=seed)
            return self._obs(), {}

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            return self._obs(), 0.0, False, True, {}

    return SAC(
        "MultiInputPolicy", _TinyGoalEnv(), buffer_size=100, learning_starts=1000, device="cpu"
    )


def test_extract_sb3_actor_dict_obs_reconstructs_dict() -> None:
    """Regression for ``Expected dict, got Tensor``: the sb3 actor wrapper must take
    a flat tensor (concat of the HER sub-spaces) and rebuild the Dict the SAC actor
    needs, matching ``policy._predict`` on the equivalent Dict."""
    from genelab.rl.backends.sb3 import _extract_sb3_actor

    model = _tiny_her_sac_model()
    actor, in_dim = _extract_sb3_actor(model)
    assert in_dim == 9  # observation(5) + achieved_goal(2) + desired_goal(2)

    flat = torch.randn(4, 9)
    with torch.no_grad():
        exported = actor(flat)
        ref = model.policy._predict(
            {
                "observation": flat[:, 0:5],
                "achieved_goal": flat[:, 5:7],
                "desired_goal": flat[:, 7:9],
            },
            deterministic=True,
        )
    assert exported.shape == (4, 3)
    assert torch.allclose(exported, ref, atol=1e-5)


@pytest.mark.parametrize("fmt", ["torchscript", "onnx"])
def test_export_sb3_dict_obs_end_to_end(tmp_path: Path, fmt: str) -> None:
    """Full goal-conditioned export: a flat-input ``.ts`` / ``.onnx`` + metadata that
    records the per-group offsets, loadable and runnable on a flat sample obs."""
    if fmt == "onnx":
        pytest.importorskip("onnx")
    from genelab.rl.backends.sb3 import _extract_sb3_actor
    from genelab.rl.exporter import ExportConfig, export_policy

    model = _tiny_her_sac_model()
    actor, in_dim = _extract_sb3_actor(model)
    out = tmp_path / f"policy.{'ts' if fmt == 'torchscript' else 'onnx'}"
    written = export_policy(
        task_id="Franka-Pick-And-Place-v0",
        env=_franka_style_env(),
        checkpoint=Path("/tmp/fake.zip"),
        actor=actor,
        actor_input_dim=in_dim,
        action_dim=3,
        policy_group="policy",
        goal_groups=["achieved_goal", "desired_goal"],
        cfg=ExportConfig(format=fmt, output=out, opset=17),
    )
    assert written == out
    assert out.exists()
    meta_path = out.with_suffix(out.suffix + ".metadata.json")
    meta = json.loads(meta_path.read_text())
    assert meta["obs_dim"] == 9
    groups = meta["obs_groups"]
    assert (groups["policy"]["start"], groups["policy"]["dim"]) == (0, 5)
    assert (groups["achieved_goal"]["start"], groups["achieved_goal"]["dim"]) == (5, 2)
    assert (groups["desired_goal"]["start"], groups["desired_goal"]["dim"]) == (7, 2)

    if fmt == "torchscript":
        loaded = torch.jit.load(str(out))
        actions = loaded(torch.zeros(6, 9))
        assert actions.shape == (6, 3)
        assert bool(torch.isfinite(actions).all())
    else:
        import onnx

        model_onnx = onnx.load(str(out))
        onnx.checker.check_model(model_onnx)
