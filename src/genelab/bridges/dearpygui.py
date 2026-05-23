"""DearPyGui twist bridge: three sliders for ``(vx, vy, ωz)`` in a side window.

Drives a 3-component velocity-twist command term (``vx``, ``vy``, ``ωz``). Not
a generic command bridge — the layout is hardcoded to that shape.

Optional extra (``uv sync --extra teleop`` installs ``dearpygui``). Lazy import
inside :py:meth:`on_build` so importing this module never crashes core when the
extra is missing — the user only pays the import cost when they actually wire
the bridge into a play config.

Threading model: DearPyGui owns its own render loop, which it expects to drive
on the main thread. Since the Genesis viewer already owns the main thread, we
run the DPG loop on a daemon thread. Slider callbacks mutate a small ``(vx, vy,
ωz)`` tuple under a ``threading.Lock``; the bridge reads under the same lock in
:py:meth:`pre_step` and broadcasts the values into the command buffer.

Caveats:

* DPG's threaded mode is supported but considered "advanced" by upstream. Some
  Linux compositors emit harmless GL warnings on the side thread.
* Closing the DPG window does NOT terminate the play loop — close the Genesis
  viewer or hit Ctrl-C. The bridge re-opens DPG state cleanly on the next play.
"""

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from genelab.bridges.base import BridgeCfg

if TYPE_CHECKING:
    from genelab.contracts import EnvContext

_logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DearPyGuiTwistBridgeCfg(BridgeCfg):
    """Configuration for :class:`DearPyGuiTwistBridge`.

    Range fields are required (no defaults): a teleop GUI cannot guess sensible
    slider limits for an unknown robot, so the caller must set them to the
    twist command term's actual range.
    """

    vx_range: tuple[float, float]
    """Slider min/max for ``vx`` (m/s). Set to your robot's twist command limits."""

    vy_range: tuple[float, float]
    """Slider min/max for ``vy`` (m/s). Set to your robot's twist command limits."""

    wz_range: tuple[float, float]
    """Slider min/max for ``ωz`` (rad/s). Set to your robot's twist command limits."""

    command_name: str = "twist"
    """Name of the command term to drive."""

    default_vx: float = 0.0
    default_vy: float = 0.0
    default_wz: float = 0.0
    """Initial slider values, also written into the command buffer at attach so
    the very first obs the policy sees is in-distribution. Defaults are zero
    (robot-agnostic safe state); override when the policy is weak at sustained
    ``[0, 0, 0]`` and you want to start in a known-good locomotion regime.
    """

    title: str = "GeneLab Teleop"
    width: int = 360
    height: int = 240

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = DearPyGuiTwistBridge


class DearPyGuiTwistBridge:
    """``Bridge`` impl that drives ``(vx, vy, ωz)`` from three DPG sliders."""

    cfg: DearPyGuiTwistBridgeCfg

    def __init__(self, cfg: DearPyGuiTwistBridgeCfg) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._vx: float = cfg.default_vx
        self._vy: float = cfg.default_vy
        self._wz: float = cfg.default_wz
        self._thread: threading.Thread | None = None
        self._dpg: Any | None = None  # populated in on_build
        self._enabled: bool = False

    # ------------------------------------------------------------------ lifecycle

    def on_build(self, env: "EnvContext") -> None:
        # Teleop only makes sense with a single env — broadcasting the slider
        # state across N envs collapses them onto the same trajectory, which
        # defeats the point of a parallel rollout. Skip silently except for a
        # single info log so the user knows what happened.
        if env.num_envs != 1:
            _logger.info(
                "DearPyGuiCommandBridge: skipping GUI (num_envs=%d, need 1). "
                "Pass --num_envs 1 to enable teleop.",
                env.num_envs,
            )
            return
        try:
            import dearpygui.dearpygui as dpg
        except ImportError as exc:
            raise ImportError(
                "DearPyGuiCommandBridge requires the optional 'teleop' extra. "
                "Install with: uv sync --extra teleop"
            ) from exc
        self._dpg = dpg

        term = env.command_manager.get_term(self.cfg.command_name)
        # Hand the command buffer over so per-step writes survive command_manager.compute.
        # Falls through gracefully on command terms that don't implement the hook.
        pin = getattr(term, "set_external_source", None)
        if callable(pin):
            pin(enabled=True)
        # Seed the command buffer with the slider defaults so the very first obs
        # the policy reads is already in-distribution. ``set_external_source``
        # zeroed it; we re-write so the term, the slider widget, and the policy
        # all agree on the same starting command.
        buf = term.command
        buf[:] = torch.tensor([self._vx, self._vy, self._wz], device=buf.device, dtype=buf.dtype)

        self._thread = threading.Thread(
            target=self._run_dpg_loop,
            name="DearPyGuiCommandBridge",
            daemon=True,
        )
        self._thread.start()
        self._enabled = True

    def pre_step(self, env: "EnvContext") -> None:
        if not self._enabled:
            return
        with self._lock:
            vx, vy, wz = self._vx, self._vy, self._wz
        term = env.command_manager.get_term(self.cfg.command_name)
        buf = term.command
        desired = torch.tensor([vx, vy, wz], device=buf.device, dtype=buf.dtype)
        buf[:] = desired

    def post_step(self, env: "EnvContext") -> None:
        pass

    def on_close(self, env: "EnvContext") -> None:
        if not self._enabled:
            return
        dpg = self._dpg
        if dpg is not None:
            try:
                dpg.stop_dearpygui()
            except Exception:
                _logger.exception("DearPyGui stop failed; ignoring during teardown")
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ slider callbacks

    def _set_vx(self, _sender: Any, value: float) -> None:
        with self._lock:
            self._vx = float(value)

    def _set_vy(self, _sender: Any, value: float) -> None:
        with self._lock:
            self._vy = float(value)

    def _set_wz(self, _sender: Any, value: float) -> None:
        with self._lock:
            self._wz = float(value)

    def _zero(self, _sender: Any = None, _app_data: Any = None) -> None:
        dpg = self._dpg
        assert dpg is not None
        with self._lock:
            self._vx = self._vy = self._wz = 0.0
        # Reset slider widget values too so the UI matches state.
        dpg.set_value("slider_vx", 0.0)
        dpg.set_value("slider_vy", 0.0)
        dpg.set_value("slider_wz", 0.0)

    # ------------------------------------------------------------------ dpg thread

    def _run_dpg_loop(self) -> None:
        dpg = self._dpg
        assert dpg is not None
        try:
            dpg.create_context()
            with dpg.window(label=self.cfg.title, tag="root", no_close=True):
                dpg.add_slider_float(
                    label="vx (m/s)",
                    tag="slider_vx",
                    min_value=self.cfg.vx_range[0],
                    max_value=self.cfg.vx_range[1],
                    default_value=self.cfg.default_vx,
                    callback=self._set_vx,
                )
                dpg.add_slider_float(
                    label="vy (m/s)",
                    tag="slider_vy",
                    min_value=self.cfg.vy_range[0],
                    max_value=self.cfg.vy_range[1],
                    default_value=self.cfg.default_vy,
                    callback=self._set_vy,
                )
                dpg.add_slider_float(
                    label="wz (rad/s)",
                    tag="slider_wz",
                    min_value=self.cfg.wz_range[0],
                    max_value=self.cfg.wz_range[1],
                    default_value=self.cfg.default_wz,
                    callback=self._set_wz,
                )
                dpg.add_button(label="Zero", callback=self._zero)
            dpg.create_viewport(
                title=self.cfg.title,
                width=self.cfg.width,
                height=self.cfg.height,
            )
            dpg.setup_dearpygui()
            dpg.set_primary_window("root", True)
            dpg.show_viewport()
            dpg.start_dearpygui()
        except Exception:
            _logger.exception("DearPyGui loop crashed; bridge will stop updating sliders")
        finally:
            try:
                dpg.destroy_context()
            except Exception:
                pass
