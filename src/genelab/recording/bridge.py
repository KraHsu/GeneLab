"""Forward-reference holder linking scene-build-time registration to env-bound runtime data.

Genesis requires ``add_recorder`` to run **before** ``gs_scene.build()``
(``@gs.assert_unbuilt``), but the data callables that recorders invoke want access to
the live :class:`~genelab.envs.manager_based_rl_env.ManagerBasedRlEnv` — which doesn't
exist until later. The :class:`RecorderBridge` is the indirection that resolves this:
closures captured at register time read ``bridge.sensors`` / ``bridge.env`` lazily, and
the env's ``__init__`` calls :meth:`bind_env` to wire the final reference once managers
are constructed and sensors are bound.

The bridge also remembers, per registered Genesis :class:`Recorder`, whether its source
is a sensor name (in which case ``_steps_per_sample`` is patched to the decimation
ratio at :meth:`bind_env`) or an opaque callable (left at every-physics-tick sampling).
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from genelab.contracts import EnvContext, SceneContext
    from genelab.sensor import Sensor


class _DropThreadExistsWarning(logging.Filter):
    """Drop Genesis's spurious ``start_thread(): Processor thread already exists`` warning.

    Genesis's ``RecorderManager.reset`` calls ``recorder.start()`` right after
    ``recorder.reset()`` without stopping the still-running processor thread, so every
    ``save_on_reset`` flush logs that warning. The restart is a harmless no-op (the thread
    keeps running), so the warning is pure noise — filtered only around our ``reset`` call.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Processor thread already exists" not in record.getMessage()


@dataclass
class RecorderHandle:
    """Pair of (Genesis recorder, default-rate policy) tracked per registration."""

    recorder: Any  # genesis.recorders.Recorder
    is_sensor_source: bool
    user_set_hz: bool
    save_on_reset: bool


class RecorderBridge:
    """Late-bound references for recording data callables.

    Lifecycle:

    1. ``__init__(scene)`` — allocated in :class:`InteractiveScene` if the scene cfg has
       any ``recordings``.
    2. ``sensors`` / ``entities`` set inside ``InteractiveScene.build`` between sensor
       pre-build and ``gs_scene.build``.
    3. ``register_recorders`` populates :attr:`handles` while building Genesis recorders.
    4. ``bind_env(env)`` called from ``ManagerBasedRlEnv.__init__`` after sensors are
       bound; sets :attr:`env` and patches the per-recorder ``_steps_per_sample`` for
       sensor-name sources to ``env._decimation``.
    5. ``on_reset(env_ids)`` invoked from ``ManagerBasedRlEnv._reset_idx`` when at least
       one recorder uses ``save_on_reset``.
    """

    def __init__(self, scene: "SceneContext") -> None:
        self.scene = scene
        self.sensors: dict[str, "Sensor[Any]"] = {}
        self.entities: dict[str, Any] = {}
        self.env: "EnvContext | None" = None
        self.handles: list[RecorderHandle] = []
        # Last Genesis global step at which we flushed recorders. Used to no-op
        # back-to-back resets (e.g. the warm-up reset in ``ManagerBasedRlEnv.__init__``
        # followed by an explicit ``env.reset()`` from the runner) so the
        # ``save_on_reset`` file counter starts at 0 instead of being prematurely
        # bumped before any data is recorded.
        self._last_reset_step: int = -1

    @property
    def has_save_on_reset(self) -> bool:
        return any(h.save_on_reset for h in self.handles)

    def bind_env(self, env: "EnvContext") -> None:
        """Wire the env reference and patch sensor-source recorder rates to control rate.

        Snapshots Genesis's current global step as the reset baseline so the warm-up
        reset triggered at the tail of ``ManagerBasedRlEnv.__init__`` is a no-op for
        ``save_on_reset`` file counters.
        """
        self.env = env
        decimation = max(1, int(env.cfg.decimation))
        for handle in self.handles:
            if handle.is_sensor_source and not handle.user_set_hz:
                handle.recorder._steps_per_sample = decimation
        self._last_reset_step = self._current_global_step(self.scene.gs_scene)

    def on_reset(self, env_ids: Any) -> None:
        """Flush + restart Genesis recorders when any has ``save_on_reset=True``.

        ``env_ids`` is ignored: file writers ignore env IDs (Genesis convention) and
        plotters squeeze to a single env upstream, so per-env partial resets aren't
        meaningful here. The call is a no-op when no simulation steps have run since
        the previous reset — that covers both the env's warm-up reset in
        ``__init__`` and any back-to-back ``env.reset()`` calls before stepping, so
        ``save_on_reset`` file counters only advance on real episode boundaries.
        """
        del env_ids
        if not self.has_save_on_reset:
            return
        gs_scene = self.scene.gs_scene
        rm = getattr(gs_scene, "_recorder_manager", None)
        if rm is None or not rm.is_recording:
            return
        cur_step = self._current_global_step(gs_scene)
        if cur_step == self._last_reset_step:
            return
        self._last_reset_step = cur_step
        # Genesis's reset() restarts already-running recorder threads, logging a harmless
        # "thread already exists" warning each episode; filter it just around this call.
        import genesis as gs

        logger = getattr(gs, "logger", None)
        log_filter = _DropThreadExistsWarning()
        if logger is not None:
            logger.addFilter(log_filter)
        try:
            rm.reset(envs_idx=None)
        finally:
            if logger is not None:
                logger.removeFilter(log_filter)

    @staticmethod
    def _current_global_step(gs_scene: Any) -> int:
        """Read Genesis's ``cur_step_global`` (the canonical recorder clock).

        Falls back to ``0`` when Genesis hasn't built yet or doesn't expose the
        attribute on this version — in that case the equality check still works as
        long as the bridge's own counter starts at ``-1``.
        """
        sim = getattr(gs_scene, "_sim", None)
        if sim is None:
            return 0
        return int(getattr(sim, "cur_step_global", 0))

    def stop(self) -> None:
        """Stop Genesis's recorder pipeline (flushes file buffers, joins threads)."""
        gs_scene = self.scene.gs_scene
        if gs_scene is None:
            return
        rm = getattr(gs_scene, "_recorder_manager", None)
        if rm is None or not rm.is_recording:
            return
        try:
            gs_scene.stop_recording()
        except Exception:  # noqa: BLE001 - cleanup must not raise
            pass
