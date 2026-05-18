"""Play-time bridges: transport-agnostic per-step hooks for `play_task`.

A `Bridge` is a small lifecycle object the play loop drives once per step. The four
hook points are enough to plug arbitrary I/O — keyboard, GUI, ROS2, gRPC, gamepad,
websocket — into a running env without forking the runner. Each bridge gets the live
``ManagerBasedRlEnv`` handle so it can read sensor / state buffers and write into
command / event manager state directly.

Bridges are configured per env: ``ManagerBasedRlEnvCfg.bridges_cfg`` is a
``dict[str, BridgeCfg]`` (sibling of ``commands_cfg``, ``events_cfg``). Only the
``play_task`` runner consumes the dict; ``train_task`` silently ignores it. Putting
bridges on the env cfg (not the task) means users who assign a bridge to
``TaskCfg.play_env`` won't accidentally carry it into training.

Timing inside the play loop:

* ``on_build`` — once, right after ``ManagerBasedRlEnv`` has finished construction.
  Safe place to grab ``env.scene.gs_scene.viewer`` and register keybinds, start a
  GUI thread, open a ROS2 / RPC connection, etc.
* ``pre_step`` — every iteration, *before* ``policy(obs)``. Bridges that drive
  commands write here (e.g. ``env.command_manager.get_term("twist")._command[:] =
  desired``). Pre-step is also the right place to consume any messages queued
  by a background thread.
* ``post_step`` — every iteration, *after* ``env.step``. Read-side bridges publish
  state / sensor data here.
* ``on_close`` — once, before ``env.close()``. Stop threads, close sockets, etc.
  Exceptions are swallowed by the runner so a misbehaving bridge can't block
  scene teardown.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@runtime_checkable
class Bridge(Protocol):
    """Lifecycle protocol implemented by every bridge.

    All four methods are optional in practice — the runner calls them via
    ``getattr(..., method, None)`` and skips missing ones — but listing them on
    the Protocol makes the contract greppable and lets type-checkers flag typos.
    """

    def on_build(self, env: "ManagerBasedRlEnv") -> None: ...
    def pre_step(self, env: "ManagerBasedRlEnv") -> None: ...
    def post_step(self, env: "ManagerBasedRlEnv") -> None: ...
    def on_close(self, env: "ManagerBasedRlEnv") -> None: ...


@dataclass
class BridgeCfg:
    """Base cfg dataclass. Subclass and set ``class_type`` to a concrete ``Bridge``.

    The runner instantiates a bridge via ``cfg.class_type(cfg)``; entries with
    ``class_type=None`` are skipped (mirrors the convention used by
    ``CommandTermCfg.class_type`` etc.).
    """

    class_type: type[Bridge] | None = None
