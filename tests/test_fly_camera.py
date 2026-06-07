"""Free-fly viewer camera math (``genelab.scene.fly_camera.FlyCamera``).

Pure pose arithmetic — no Genesis, no viewer — so the WASD/mouse fly controls can be
verified deterministically. World frame is Z-up (Genesis convention): yaw rotates about
+Z from +X, pitch tilts up (+) / down (-).
"""

import math

import numpy as np

from genelab.scene.fly_camera import FlyCamera, FlyCameraController, purge_held_key_symbol


def test_purge_held_key_symbol_drops_every_modifier_variant() -> None:
    # Reproduces the sticky-Shift fly bug: pyrender keys _held_keys by (symbol, modifiers),
    # but Shift toggles its own modifier bit, so its press records (LSHIFT, 0) while its
    # release reports (LSHIFT, MOD_SHIFT). A naive pop of the release tuple misses, the
    # (LSHIFT, 0) entry lingers, and the HOLD "down" callback fires forever. Releasing a key
    # means it's up regardless of modifier state, so all of its entries must go.
    LSHIFT, SPACE, MOD_SHIFT = 65505, 32, 1
    held = {(LSHIFT, 0): True, (LSHIFT, MOD_SHIFT): True, (SPACE, 0): True}
    purge_held_key_symbol(held, LSHIFT)
    assert not any(sym == LSHIFT for (sym, _mods) in held)
    assert (SPACE, 0) in held  # unrelated keys are untouched


def test_purge_held_key_symbol_noop_when_symbol_absent() -> None:
    held = {(32, 0): True}
    purge_held_key_symbol(held, 99)
    assert held == {(32, 0): True}


def _dir(pos, lookat) -> np.ndarray:
    v = np.asarray(lookat, float) - np.asarray(pos, float)
    return v / np.linalg.norm(v)


def test_from_look_round_trips_the_view_direction() -> None:
    cam = FlyCamera.from_look(pos=(0.0, 0.0, 1.0), lookat=(2.0, 0.0, 1.0))
    pos, lookat = cam.pose()
    assert np.allclose(pos, (0.0, 0.0, 1.0), atol=1e-6)
    assert np.allclose(_dir(pos, lookat), (1.0, 0.0, 0.0), atol=1e-6)


def test_move_forward_advances_along_view_direction() -> None:
    cam = FlyCamera.from_look(pos=(0.0, 0.0, 1.0), lookat=(1.0, 0.0, 1.0))  # facing +X
    cam.move(forward=2.0)
    pos, _ = cam.pose()
    assert np.allclose(pos, (2.0, 0.0, 1.0), atol=1e-6)


def test_look_yaws_the_view_and_clamps_pitch() -> None:
    cam = FlyCamera.from_look(pos=(0.0, 0.0, 0.0), lookat=(1.0, 0.0, 0.0))  # facing +X
    cam.look(dyaw=math.pi / 2)  # quarter turn left → faces +Y
    assert np.allclose(cam.forward, (0.0, 1.0, 0.0), atol=1e-6)
    # Pitching far past vertical clamps just shy of straight up (no horizon flip).
    cam.look(dpitch=10.0)
    assert cam.forward[2] < 1.0  # never reaches a fully vertical, degenerate view
    assert math.isclose(float(np.linalg.norm(cam.forward)), 1.0, abs_tol=1e-6)


def test_strafe_stays_level_and_up_is_world_z() -> None:
    # Facing +X and pitched down: strafing right must stay in the XY plane (no height change),
    # and Space/Shift (up) must move along world +Z regardless of the downward look.
    cam = FlyCamera.from_look(pos=(0.0, 0.0, 5.0), lookat=(1.0, 0.0, 4.0))  # facing +X, tilted down
    cam.move(right=2.0)
    pos, _ = cam.pose()
    assert np.allclose(pos, (0.0, -2.0, 5.0), atol=1e-6)  # right of +X is -Y, z unchanged
    cam.move(up=3.0)
    pos, _ = cam.pose()
    assert np.allclose(pos, (0.0, -2.0, 8.0), atol=1e-6)  # straight up despite looking down


def test_controller_only_moves_while_fly_mode_is_active() -> None:
    # Fly mode is gated on holding the right mouse button. Movement / look inputs are ignored
    # until activated, so normal orbit controls are untouched when the button isn't held.
    cam = FlyCamera.from_look(pos=(0.0, 0.0, 0.0), lookat=(1.0, 0.0, 0.0))
    ctrl = FlyCameraController(cam, move_speed=3.0)

    ctrl.move("forward", dt=1.0)  # not active → ignored
    assert np.allclose(cam.pose()[0], (0.0, 0.0, 0.0), atol=1e-6)

    ctrl.set_active(True)
    ctrl.move("forward", dt=1.0)  # active → advances at move_speed
    assert np.allclose(cam.pose()[0], (3.0, 0.0, 0.0), atol=1e-6)


def test_reseat_starts_fly_from_the_current_view() -> None:
    # When fly mode engages, it must continue from wherever the user orbited to, not the
    # camera's build-time pose — so the wiring reseats it from the live viewer pose.
    ctrl = FlyCameraController(FlyCamera.from_look((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    ctrl.reseat(pos=(5.0, 5.0, 2.0), lookat=(5.0, 6.0, 2.0))  # now at (5,5,2) facing +Y
    pos, lookat = ctrl.pose()
    assert np.allclose(pos, (5.0, 5.0, 2.0), atol=1e-6)
    assert np.allclose((lookat - pos) / np.linalg.norm(lookat - pos), (0.0, 1.0, 0.0), atol=1e-6)
