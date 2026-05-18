"""Tests for the play-time bridge protocol and reference implementations."""

from dataclasses import dataclass, field
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.bridges import (  # noqa: E402 (after importorskip)
    Bridge,
    BridgeCfg,
    KeyboardCommandBridge,
    KeyboardCommandBridgeCfg,
)
from genelab.configs import apply_overrides  # noqa: E402
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg  # noqa: E402
from genelab.rl.runner import _build_bridges, _close_bridges, _run_play_loop  # noqa: E402


# ---------------------------------------------------------------------- fakes


class _RecordingBridge:
    """Bridge stub that appends a tag to a shared log on every hook call."""

    def __init__(self, log: list[str], name: str = "rec") -> None:
        self._log = log
        self._name = name

    def on_build(self, _env: Any) -> None:
        self._log.append(f"{self._name}.on_build")

    def pre_step(self, _env: Any) -> None:
        self._log.append(f"{self._name}.pre_step")

    def post_step(self, _env: Any) -> None:
        self._log.append(f"{self._name}.post_step")

    def on_close(self, _env: Any) -> None:
        self._log.append(f"{self._name}.on_close")


@dataclass
class _RecordingBridgeCfg(BridgeCfg):
    log: list[str] = field(default_factory=list)
    name: str = "rec"
    class_type: type[Bridge] | None = None

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = _make_recording_bridge


def _make_recording_bridge(cfg: _RecordingBridgeCfg) -> _RecordingBridge:
    return _RecordingBridge(cfg.log, cfg.name)


class _FakeWrapped:
    """Stand-in for ``RslRlVecEnvWrapper`` — only the methods the play loop calls."""

    def __init__(self) -> None:
        self._obs = torch.zeros(1, 4)

    def reset(self) -> tuple[Any, dict[str, Any]]:
        return self._obs, {}

    def step(self, _actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        return self._obs, torch.zeros(1), torch.zeros(1, dtype=torch.bool), {}


class _FakeEnv:
    """Stand-in for ``ManagerBasedRlEnv``. Only carries the attributes the loop reads."""

    def __init__(self) -> None:
        self.viewer_closed = False


def _zero_policy(_obs: Any) -> Any:
    return torch.zeros(1, 4)


# ---------------------------------------------------------------------- tests


def test_run_play_loop_calls_bridges_in_order() -> None:
    """The contract the runner pins: pre_step → policy/step → post_step, repeated."""
    log: list[str] = []
    bridge = _RecordingBridge(log)
    env = _FakeEnv()
    wrapped = _FakeWrapped()

    _run_play_loop(env, wrapped, _zero_policy, [bridge], max_steps=2, prof_step=None)
    # on_build / on_close are runner-level (outside _run_play_loop) — caller is
    # responsible. Loop itself drives only pre_step / post_step.
    assert log == [
        "rec.pre_step",
        "rec.post_step",
        "rec.pre_step",
        "rec.post_step",
    ]


def test_run_play_loop_breaks_on_viewer_closed() -> None:
    log: list[str] = []
    env = _FakeEnv()
    wrapped = _FakeWrapped()

    class _ClosingBridge:
        def on_build(self, _env: Any) -> None: ...
        def pre_step(self, _env: Any) -> None:
            log.append("pre")

        def post_step(self, _env: Any) -> None:
            log.append("post")
            env.viewer_closed = True  # flip mid-loop

        def on_close(self, _env: Any) -> None: ...

    _run_play_loop(env, wrapped, _zero_policy, [_ClosingBridge()], max_steps=None, prof_step=None)
    # One full iteration then break before pre_step of iteration 2.
    assert log == ["pre", "post"]


def test_build_bridges_skips_none_class_type() -> None:
    log: list[str] = []
    cfg = ManagerBasedRlEnvCfg()
    cfg.bridges_cfg = {
        "real": _RecordingBridgeCfg(log=log, name="real"),
        "ghost": BridgeCfg(class_type=None),
    }
    bridges = _build_bridges(cfg)
    assert len(bridges) == 1
    bridges[0].on_build(_FakeEnv())
    assert log == ["real.on_build"]


def test_close_bridges_swallows_per_bridge_exceptions() -> None:
    """A misbehaving on_close must not block teardown of subsequent bridges."""

    class _BadBridge:
        def on_build(self, _env: Any) -> None: ...
        def pre_step(self, _env: Any) -> None: ...
        def post_step(self, _env: Any) -> None: ...
        def on_close(self, _env: Any) -> None:
            raise RuntimeError("boom")

    log: list[str] = []
    good = _RecordingBridge(log, name="good")
    _close_bridges([_BadBridge(), good], _FakeEnv())
    assert log == ["good.on_close"]


def test_bridges_cfg_round_trips_through_apply_overrides() -> None:
    """``bridges_cfg.<name>.<field>`` resolves the same way as commands_cfg etc."""
    cfg = ManagerBasedRlEnvCfg()
    cfg.bridges_cfg = {"teleop": KeyboardCommandBridgeCfg()}
    apply_overrides(cfg, {"bridges_cfg.teleop.command_name": "my_twist"})
    apply_overrides(cfg, {"bridges_cfg.teleop.step_lin": "0.25"})
    teleop = cfg.bridges_cfg["teleop"]
    assert isinstance(teleop, KeyboardCommandBridgeCfg)
    assert teleop.command_name == "my_twist"
    assert teleop.step_lin == pytest.approx(0.25)


def test_keyboard_bridge_skips_when_num_envs_above_one() -> None:
    """Teleop is num_envs=1-only: anything else short-circuits at on_build.

    Verifies the bridge degrades safely when attached to a multi-env play (e.g.
    default Unitree velocity play with num_envs=50).
    """
    bridge = KeyboardCommandBridge(KeyboardCommandBridgeCfg())

    class _MultiEnv:
        num_envs = 50

    bridge.on_build(_MultiEnv())  # type: ignore[arg-type]
    # pre_step must short-circuit (no command_manager to write into).
    bridge.pre_step(_MultiEnv())  # type: ignore[arg-type]


def test_keyboard_bridge_no_viewer_is_a_noop() -> None:
    """num_envs=1 but vis=False → bridge stays disabled without crashing."""
    bridge = KeyboardCommandBridge(KeyboardCommandBridgeCfg())

    class _SimCfg:
        vis = False

    class _Cfg:
        simulation = _SimCfg()

    class _HeadlessEnv:
        num_envs = 1
        cfg = _Cfg()

    bridge.on_build(_HeadlessEnv())  # type: ignore[arg-type]
    bridge.pre_step(_HeadlessEnv())  # type: ignore[arg-type]
