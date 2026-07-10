"""Keyboard twist bridge: drive a 3-component (vx, vy, ωz) command term from WASD / QE / Space.

Drives a 3-component velocity-twist command term — the layout is hardcoded to
that shape (not a generic command bridge). It is the simplest possible reference
implementation of the :class:`Bridge` protocol, using Genesis's built-in keybind
mechanism (``Viewer.register_keybinds``) so no extra dependency is needed.

Default keymap (rebind via cfg fields):

* ``W`` / ``S`` — increment / decrement ``vx`` by ``step_lin``.
* ``A`` / ``D`` — increment / decrement ``vy`` by ``step_lin``.
* ``Q`` / ``E`` — increment / decrement ``ωz`` by ``step_ang``.
* ``SPACE`` — zero the command.

The bridge writes into ``env.command_manager.get_term(command_name)._command``
in :py:meth:`pre_step`, so its value survives the next ``command_manager.compute``
as long as random resampling is disabled on the term (set
``rel_standing_envs = rel_heading_envs = rel_forward_envs = rel_world_envs = 0``,
``heading_command = False``, and ``resampling_time_range`` to something huge —
``(1e9, 1e9)`` is the established play idiom).

When ``simulation.vis`` is ``False`` the bridge logs a warning at ``on_build`` and
turns itself into a no-op: there is no viewer to receive keybinds.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.bridges.base import BridgeCfg

if TYPE_CHECKING:
    from genelab.contracts import EnvContext

_logger = logging.getLogger(__name__)


@dataclass
class KeyboardTwistBridgeCfg(BridgeCfg):
    """Configuration for :class:`KeyboardTwistBridge`."""

    command_name: str = "twist"
    """Name of the command term to drive (e.g. the velocity ``twist`` term)."""

    step_lin: float = 0.1
    """Per-keypress increment for ``vx`` / ``vy`` (m/s)."""

    step_ang: float = 0.2
    """Per-keypress increment for ``ωz`` (rad/s)."""

    show_hud: bool = True
    """Echo the current ``(vx, vy, ωz)`` on the Genesis viewer's HUD line."""

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = KeyboardTwistBridge


class KeyboardTwistBridge:
    """``Bridge`` impl that maps keyboard input → velocity command buffer."""

    cfg: KeyboardTwistBridgeCfg

    def __init__(self, cfg: KeyboardTwistBridgeCfg) -> None:
        self.cfg = cfg
        # Per-axis desired command; written by key callbacks, read in pre_step.
        # Plain Python floats — single-threaded mutation from pyglet's event
        # handler so no lock is needed.
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._wz: float = 0.0
        self._enabled: bool = False

    # ------------------------------------------------------------------ lifecycle

    def on_build(self, env: "EnvContext") -> None:
        if env.num_envs != 1:
            _logger.info(
                "KeyboardTwistBridge: skipping (num_envs=%d, need 1). "
                "Pass --num_envs 1 to enable teleop.",
                env.num_envs,
            )
            return
        sim_cfg = getattr(env.cfg, "simulation", None)
        if sim_cfg is None or not getattr(sim_cfg, "vis", False):
            _logger.warning(
                "KeyboardTwistBridge: simulation.vis is False — no viewer to attach "
                "keybinds to. Bridge will be a no-op."
            )
            return
        viewer = env.scene.gs_scene.viewer
        if viewer is None:
            _logger.warning("KeyboardTwistBridge: scene has no viewer; bridge disabled.")
            return

        term = env.command_manager.get_term(self.cfg.command_name)
        pin = getattr(term, "set_external_source", None)
        if callable(pin):
            pin(enabled=True)

        from genesis.vis.keybindings import Key, KeyAction, Keybind

        viewer.register_keybinds(
            Keybind(
                "teleop_vx_plus",
                Key.W,
                key_action=KeyAction.PRESS,
                callback=lambda: self._bump(0, +self.cfg.step_lin, viewer),
            ),
            Keybind(
                "teleop_vx_minus",
                Key.S,
                key_action=KeyAction.PRESS,
                callback=lambda: self._bump(0, -self.cfg.step_lin, viewer),
            ),
            Keybind(
                "teleop_vy_plus",
                Key.A,
                key_action=KeyAction.PRESS,
                callback=lambda: self._bump(1, +self.cfg.step_lin, viewer),
            ),
            Keybind(
                "teleop_vy_minus",
                Key.D,
                key_action=KeyAction.PRESS,
                callback=lambda: self._bump(1, -self.cfg.step_lin, viewer),
            ),
            Keybind(
                "teleop_wz_plus",
                Key.Q,
                key_action=KeyAction.PRESS,
                callback=lambda: self._bump(2, +self.cfg.step_ang, viewer),
            ),
            Keybind(
                "teleop_wz_minus",
                Key.E,
                key_action=KeyAction.PRESS,
                callback=lambda: self._bump(2, -self.cfg.step_ang, viewer),
            ),
            Keybind(
                "teleop_zero",
                Key.SPACE,
                key_action=KeyAction.PRESS,
                callback=lambda: self._zero(viewer),
            ),
        )
        self._enabled = True
        self._broadcast_hud(viewer)

    def pre_step(self, env: "EnvContext") -> None:
        if not self._enabled:
            return
        term = env.command_manager.get_term(self.cfg.command_name)
        buf = term.command
        # Build the (3,) tensor on the term's device, then broadcast across envs.
        desired = torch.tensor([self._vx, self._vy, self._wz], device=buf.device, dtype=buf.dtype)
        buf[:] = desired

    def post_step(self, env: "EnvContext") -> None:
        # Read-side teleop has no output stream; subclass if you want one.
        pass

    def on_close(self, env: "EnvContext") -> None:
        # Genesis tears the viewer down with the scene; nothing to release.
        pass

    # ------------------------------------------------------------------ internals

    def _bump(self, axis: int, delta: float, viewer: object) -> None:
        if axis == 0:
            self._vx += delta
        elif axis == 1:
            self._vy += delta
        else:
            self._wz += delta
        self._broadcast_hud(viewer)

    def _zero(self, viewer: object) -> None:
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._broadcast_hud(viewer)

    def _broadcast_hud(self, viewer: object) -> None:
        if not self.cfg.show_hud:
            return
        msg = f"teleop  vx={self._vx:+.2f}  vy={self._vy:+.2f}  wz={self._wz:+.2f}"
        # Genesis 1.2's outer Viewer forwards register_keybinds but not set_message_text —
        # the HUD line lives on the inner pyrender viewer (same wrapper gap as the
        # viewer.plugins / _viewer_plugins fallback in scene).
        set_text = getattr(viewer, "set_message_text", None)
        if set_text is None:
            inner = getattr(viewer, "_pyrender_viewer", None)
            set_text = getattr(inner, "set_message_text", None)
        if set_text is not None:
            set_text(msg)
