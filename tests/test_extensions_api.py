"""Public extension API surface.

``genelab.extensions`` is the single import path third parties use to register the
four extension kinds (robot / env / task / RL backend) and to reference the
``Runnable`` and ``Backend`` Protocols. These tests round-trip register-and-lookup
for each kind via the public path and check the re-exports are the real objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import genelab.extensions as ext
from genelab.registry import ENVS, ROBOTS, TASKS


def test_extensions_reexports_are_the_real_objects() -> None:
    import genelab.registry as registry
    from genelab.rl import backends

    assert ext.register_robot is registry.register_robot
    assert ext.register_env is registry.register_env
    assert ext.register_task is registry.register_task
    assert ext.Runnable is registry.Runnable
    assert ext.register_backend is backends.register_backend
    assert ext.Backend is backends.Backend
    assert ext.ROBOTS is registry.ROBOTS


def test_register_robot_env_task_round_trip() -> None:
    sentinel_robot = object()
    sentinel_env = object()
    sentinel_task = object()

    ext.register_robot("ExtTest-Robot", lambda: sentinel_robot, description="test robot")
    ext.register_env("ExtTest-Env", lambda: sentinel_env, description="test env")
    ext.register_task("ExtTest-Task", lambda: sentinel_task, description="test task")

    assert ROBOTS.get("ExtTest-Robot") is sentinel_robot
    assert ENVS.get("ExtTest-Env") is sentinel_env
    assert TASKS.get("ExtTest-Task") is sentinel_task


class _ExtTestAgentCfg:
    """Unique agent-cfg type so the fake backend keys cleanly into the registry."""


class _ExtTestBackend:
    name = "ext-test-backend"
    cfg_type = _ExtTestAgentCfg

    def train(self, ctx: Any) -> Path:  # pragma: no cover - never invoked
        raise NotImplementedError

    def play(self, ctx: Any) -> None:  # pragma: no cover - never invoked
        raise NotImplementedError

    def make_inference_setup(self, ctx: Any) -> Any:  # pragma: no cover - never invoked
        raise NotImplementedError


def test_register_backend_round_trip() -> None:
    from genelab.rl import backends

    ext.register_backend(_ExtTestBackend())
    try:
        # select_backend imports the bundled backend *modules* (optional-dep-safe;
        # they don't import the cv2-pulling vecenv wrappers at module load), then
        # resolves by agent-cfg type.
        resolved = backends.select_backend(_ExtTestAgentCfg())
        assert resolved.name == "ext-test-backend"
    finally:
        backends._BACKENDS.pop(_ExtTestAgentCfg, None)  # pyright: ignore[reportPrivateUsage]


def test_runnable_protocol_shape() -> None:
    # A class with the right shape is a structural ``Runnable``; this documents the
    # contract third-party task types follow (cfg + play(max_steps=...) + train()).
    # ``play`` takes the keyword-only ``--max-steps`` hard cap; tasks that drive no
    # step loop may accept and ignore it.
    class _OkTask:
        cfg = object()

        def play(self, *, max_steps: int | None = None) -> None: ...

        def train(self) -> None: ...

    task: ext.Runnable = _OkTask()
    assert hasattr(task, "cfg") and callable(task.play) and callable(task.train)
