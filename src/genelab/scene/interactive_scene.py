"""Composite Genesis scene: groups articulations + rigid objects + ground + mouse plugin.

The env hands two cfgs to :class:`InteractiveScene`: :class:`SimulationCfg` (Genesis runtime
options) and :class:`InteractiveSceneCfg` (scene composition). The scene owns the Genesis
``gs.Scene`` handle and the entity wrappers; the env owns the managers.

Lifecycle:

1. ``InteractiveScene(sim_cfg, scene_cfg, device_hint=...)`` allocates wrappers but does NOT
   touch Genesis yet.
2. ``add_entity(name, cfg)`` registers extra entities before build (the env uses this to
   inject ``cfg.robot`` from ``ManagerBasedRlEnvCfg``).
3. ``build()`` initializes Genesis, constructs the parallel scene, spawns entities, attaches
   the mouse plugin if requested, calls ``scene.build(...)``, then binds every articulation
   so its post-build introspection lands.
"""

import logging
import os
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import torch

from genelab.entity import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from genelab.entity._torch import to_tensor
from genelab.recording.bridge import RecorderBridge
from genelab.sensor import Sensor
from genelab.terrains import TerrainImporter

if TYPE_CHECKING:
    from genelab.configs import SimulationCfg
    from genelab.sensor.camera import CameraSensor


_logger = logging.getLogger(__name__)


def _viewer_option_kwargs(sim_cfg: "SimulationCfg", *, enable_gui: bool) -> dict[str, Any]:
    """Assemble the ``gs.options.ViewerOptions`` kwargs from a :class:`SimulationCfg`.

    Camera framing is only forwarded when set so an unconfigured scene keeps Genesis' own
    default camera; ``camera_lookat`` doubles as the trackball pivot, so framing the subject
    is what makes the mouse-wheel zoom close in on it.
    """
    kwargs: dict[str, Any] = {"max_FPS": sim_cfg.render_fps, "enable_gui": enable_gui}
    if sim_cfg.camera_pos is not None:
        kwargs["camera_pos"] = sim_cfg.camera_pos
    if sim_cfg.camera_lookat is not None:
        kwargs["camera_lookat"] = sim_cfg.camera_lookat
    if sim_cfg.fly_camera:
        # Genesis' default controls bind W/A/S/D (world-frame / camera-rotation / save-image /
        # wireframe), which collide with fly-mode WASD. Genesis only auto-disables them when the
        # ImGui overlay is on; turning them off here lets fly own the keyboard either way.
        kwargs["enable_default_keybinds"] = False
    return kwargs


_pyrender_save_patch_applied = False
_imgui_overlay_patch_applied = False
_viewer_title_patch_applied = False


def find_imgui_panel_host(viewer: Any) -> Any | None:
    """Return the viewer plugin exposing ``register_panel`` (Genesis's ImGui overlay), or None.

    Genesis appends an ``ImGuiOverlayPlugin`` to ``viewer._viewer_plugins`` when the scene is
    built with ``ViewerOptions(enable_gui=True)``. We duck-type rather than import the plugin
    class so this stays robust across Genesis point releases — any plugin that exposes a
    ``register_panel`` callable is a valid host.
    """
    for plugin in getattr(viewer, "_viewer_plugins", None) or []:
        if callable(getattr(plugin, "register_panel", None)):
            return plugin
    return None


def register_viewer_panels(
    viewer: Any,
    panels: Iterable[Callable[[Any], None]],
    *,
    section: str = "side",
    logger: logging.Logger | None = None,
) -> int:
    """Forward each ``panel(imgui)`` callback to the viewer's ImGui overlay.

    This is the whole of GeneLab's "GUI wrapping": a user appends a plain
    ``callback(imgui)`` to :attr:`SimulationCfg.panels` and GeneLab calls
    :meth:`ImGuiOverlayPlugin.register_panel` for them after ``scene.build`` — so adding a
    GUI element costs exactly one function, same as raw Genesis. Returns the number of
    panels registered; logs a warning and returns ``0`` when the viewer has no ImGui overlay
    (i.e. the scene was built without ``enable_gui=True``).
    """
    panel_list = list(panels)
    if not panel_list:
        return 0
    host = find_imgui_panel_host(viewer)
    # Any overlay user (managed InteractiveScene or a bespoke ``gs.Scene``) gets the
    # overlay close-crash / reversed-scroll fixes for free by going through this helper.
    _patch_imgui_overlay()
    if host is None:
        if logger is not None:
            logger.warning(
                "register_viewer_panels: %d panel(s) requested but the viewer has no ImGui "
                "overlay (build the scene with viewer_imgui=True / enable_gui=True); ignored.",
                len(panel_list),
            )
        return 0
    for panel in panel_list:
        host.register_panel(panel, section=section)
    return len(panel_list)


def _patch_pyrender_save_filename() -> None:
    """Coerce pyrender's tkinter SaveAs default extension to match the requested one.

    Upstream pyrender (still in Genesis 1.0) hard-codes ``defaultextension=".png"`` in
    ``_get_save_filename`` regardless of ``file_exts``, so pressing ``R`` twice to save a
    recorded video (called with ``file_exts=["mp4"]``) and typing a bare filename produces
    e.g. ``dancing.png`` — and ``Viewer.save_video`` then ``shutil.move``s the .mp4 file
    to that .png path. We can't modify Genesis, so we wrap the method at the class level:
    when exactly one extension is requested and the dialog returned a different one, swap
    the extension. Multi-extension dialogs (e.g. ``_save_image`` offers png/jpg/gif/all)
    are left alone so the user's chosen type is respected.
    """
    global _pyrender_save_patch_applied
    if _pyrender_save_patch_applied:
        return
    try:
        from genesis.ext.pyrender.viewer import Viewer as _PyrenderViewer
    except Exception:
        return
    original = _PyrenderViewer._get_save_filename  # pyright: ignore[reportPrivateUsage]

    def _genelab_get_save_filename(self: Any, file_exts: list[str]) -> str | None:
        filename = original(self, file_exts)
        if filename is None or len(file_exts) != 1:
            return filename
        expected_ext = f".{file_exts[0]}".lower()
        base, ext = os.path.splitext(filename)
        if ext.lower() != expected_ext:
            filename = base + expected_ext
        return filename

    _PyrenderViewer._get_save_filename = _genelab_get_save_filename  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]
    _pyrender_save_patch_applied = True


def _patch_imgui_overlay() -> None:
    """Work around two Genesis ``ImGuiOverlayPlugin`` bugs that surface once the overlay is on.

    1. **Close crash.** ``on_close`` calls ``self._impl.shutdown()`` →
       ``imgui.get_platform_io()``, which on the pyglet / ``imgui_bundle`` backend asserts
       ``No current context`` when the window closes (the imgui context is already gone on the
       calling thread). The viewer thread stores that as ``_exception``; the main thread's
       ``is_alive()`` re-raises it as ``GenesisException("Unexpected viewer error.")``, so
       ``play`` exits with a traceback even though the viewer rendered fine. We wrap
       ``on_close`` to swallow teardown errors — the viewer is closing, so any leaked GL /
       imgui state is freed at process exit and the normal ``"Viewer closed."`` path takes over.

    2. **Reversed scroll.** Genesis forwards pyglet's ``scroll_y`` straight to the backend,
       which negates it (``add_mouse_wheel_event(0, -scroll)``) — so the wheel scrolls overlay
       panels the wrong way. We wrap ``on_mouse_scroll`` to flip ``dy``, restoring the expected
       direction (panel scroll only; camera zoom is a separate handler).

    We can't modify Genesis, so both are class-level wraps (mirrors
    :func:`_patch_pyrender_save_filename`). Idempotent; only matters when ``viewer_imgui`` /
    ``panels`` enabled the overlay.
    """
    global _imgui_overlay_patch_applied
    if _imgui_overlay_patch_applied:
        return
    try:
        from genesis.ext.pyrender.overlay.plugin import ImGuiOverlayPlugin
    except Exception:
        return

    original_on_close = ImGuiOverlayPlugin.on_close

    def _genelab_safe_on_close(self: Any) -> None:
        try:
            original_on_close(self)
        except Exception:
            _logger.debug(
                "ImGui overlay on_close teardown failed; ignoring on viewer close",
                exc_info=True,
            )

    original_on_scroll = ImGuiOverlayPlugin.on_mouse_scroll

    def _genelab_on_mouse_scroll(self: Any, x: Any, y: Any, dx: Any, dy: Any) -> Any:
        return original_on_scroll(self, x, y, dx, -dy)

    ImGuiOverlayPlugin.on_close = _genelab_safe_on_close  # type: ignore[method-assign]
    ImGuiOverlayPlugin.on_mouse_scroll = _genelab_on_mouse_scroll  # type: ignore[method-assign]
    _imgui_overlay_patch_applied = True


def _patch_viewer_window_title() -> None:
    """Rebrand the viewer's OS window caption from ``Genesis <ver>`` to ``GeneLab <ver>``.

    Genesis hard-codes ``viewer_flags={"window_title": f"Genesis {gs.__version__}"}`` when it
    constructs the pyrender viewer (``genesis/vis/viewer.py``), and the pyrender ``Viewer``
    applies that flag as the window caption during ``__init__``. We can't modify Genesis, so we
    wrap the pyrender ``Viewer.__init__`` at the class level (mirrors
    :func:`_patch_pyrender_save_filename`) and rewrite the incoming ``window_title`` before the
    original runs — only when it carries the ``Genesis`` default, so a caller-supplied title is
    left untouched. Idempotent; only matters when the viewer is enabled.
    """
    global _viewer_title_patch_applied
    if _viewer_title_patch_applied:
        return
    try:
        from genesis.ext.pyrender.viewer import Viewer as _PyrenderViewer

        from genelab import __version__ as _genelab_version
    except Exception:
        return
    original_init = _PyrenderViewer.__init__

    def _genelab_viewer_init(self: Any, *args: Any, **kwargs: Any) -> None:
        viewer_flags = kwargs.get("viewer_flags")
        if isinstance(viewer_flags, dict):
            title = viewer_flags.get("window_title")
            if isinstance(title, str) and title.startswith("Genesis"):
                viewer_flags["window_title"] = f"GeneLab {_genelab_version}"
        original_init(self, *args, **kwargs)

    _PyrenderViewer.__init__ = _genelab_viewer_init  # type: ignore[method-assign]
    _viewer_title_patch_applied = True


_fly_mouse_patch_applied = False


def _patch_viewer_fly_mouse() -> None:
    """Route right-mouse-drag to a per-viewer free-look fly controller, bypassing trackball zoom.

    The pyrender Viewer binds the right button to trackball zoom-drag. When a fly handle is
    attached to a viewer instance (``_genelab_fly``, set by :func:`_install_fly_camera`), the
    right button instead drives Blender-style free-look: press engages fly mode (seeded from the
    current view), drag rotates, release returns to orbit. Viewers without the handle keep stock
    behaviour. Class-level wrap, idempotent; mirrors :func:`_patch_pyrender_save_filename`.

    Also wraps ``on_key_release`` to fix a pyrender held-keys bug (see
    :func:`~genelab.scene.fly_camera.purge_held_key_symbol`) where a modifier key bound to a
    ``HOLD`` action (fly "down" on Shift) never registers its release and descends forever.
    """
    global _fly_mouse_patch_applied
    if _fly_mouse_patch_applied:
        return
    try:
        from genesis.ext.pyrender.viewer import Viewer as _PyrenderViewer
        from pyglet.window import mouse as _mouse

        from genelab.scene.fly_camera import purge_held_key_symbol
    except Exception:
        return

    orig_press = _PyrenderViewer.on_mouse_press
    orig_drag = _PyrenderViewer.on_mouse_drag
    orig_release = _PyrenderViewer.on_mouse_release
    orig_key_release = _PyrenderViewer.on_key_release
    orig_motion = _PyrenderViewer.on_mouse_motion

    def _grab_mouse(self: Any, grab: bool) -> None:
        # Lock the cursor while flying so free-look uses unbounded relative motion: without
        # this the pointer hits the screen edge and rotation stalls. Guarded — exclusive
        # mouse can be unavailable on some backends / headless windows.
        try:
            self.set_exclusive_mouse(bool(grab))
        except Exception:
            pass

    def on_mouse_press(self: Any, x: int, y: int, button: int, modifiers: int) -> Any:
        fly = getattr(self, "_genelab_fly", None)
        if fly is not None and button == _mouse.RIGHT:
            fly.engage()
            _grab_mouse(self, True)
            return None
        return orig_press(self, x, y, button, modifiers)

    def on_mouse_drag(
        self: Any, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int
    ) -> Any:
        fly = getattr(self, "_genelab_fly", None)
        if fly is not None and (buttons & _mouse.RIGHT) and fly.controller.active:
            fly.look(dx, dy)
            return None
        return orig_drag(self, x, y, dx, dy, buttons, modifiers)

    def on_mouse_motion(self: Any, x: int, y: int, dx: int, dy: int) -> Any:
        # In exclusive-mouse (fly) mode pyglet delivers relative motion via on_mouse_motion
        # rather than on_mouse_drag, so drive free-look from here too while fly is active.
        fly = getattr(self, "_genelab_fly", None)
        if fly is not None and fly.controller.active:
            fly.look(dx, dy)
            return None
        return orig_motion(self, x, y, dx, dy)

    def on_mouse_release(self: Any, x: int, y: int, button: int, modifiers: int) -> Any:
        fly = getattr(self, "_genelab_fly", None)
        if fly is not None and button == _mouse.RIGHT:
            fly.controller.set_active(False)
            _grab_mouse(self, False)  # release the cursor lock when fly ends
            return None
        return orig_release(self, x, y, button, modifiers)

    def on_key_release(self: Any, symbol: int, modifiers: int) -> Any:
        # Fix the pyrender sticky-modifier bug before delegating: a modifier key (Shift)
        # reports different modifiers on press vs release, so the base handler's
        # ``pop((symbol, modifiers))`` misses and its HOLD callback (fly "down") never stops.
        # Purge every held entry for this symbol so release is always honoured.
        held = getattr(self, "_held_keys", None)
        if isinstance(held, dict):
            purge_held_key_symbol(held, symbol)
        return orig_key_release(self, symbol, modifiers)

    _PyrenderViewer.on_mouse_press = on_mouse_press  # type: ignore[method-assign]
    _PyrenderViewer.on_mouse_drag = on_mouse_drag  # type: ignore[method-assign]
    _PyrenderViewer.on_mouse_motion = on_mouse_motion  # type: ignore[method-assign]
    _PyrenderViewer.on_mouse_release = on_mouse_release  # type: ignore[method-assign]
    _PyrenderViewer.on_key_release = on_key_release  # type: ignore[method-assign]
    _fly_mouse_patch_applied = True


def _install_fly_camera(gs_viewer: Any, sim_cfg: "SimulationCfg") -> None:
    """Attach a :class:`FlyCameraController` to a built Genesis viewer.

    Registers HOLD keybinds (W/A/S/D + Space/Shift) that fly while the right button is held, and
    a ``_genelab_fly`` handle the patched mouse handlers (:func:`_patch_viewer_fly_mouse`) drive
    for engage / free-look. Everything is guarded so a viewer that can't host it stays untouched.
    """
    try:
        import time

        import numpy as np

        from genesis.vis.keybindings import Key, KeyAction, Keybind

        from genelab.scene.fly_camera import FlyCamera, FlyCameraController
    except Exception:
        return
    pyrender_viewer = getattr(gs_viewer, "_pyrender_viewer", None)
    if pyrender_viewer is None:
        return

    controller = FlyCameraController(
        FlyCamera.from_look((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        move_speed=float(sim_cfg.fly_camera_speed),
    )

    def _upright_pose(eye: np.ndarray, lookat: np.ndarray) -> np.ndarray:
        """4x4 camera-to-world pose framed at ``lookat`` with world-Z up — i.e. zero roll.

        Building the pose ourselves (instead of ``set_camera_pose(pos, lookat)``, which carries
        the previous, possibly drifted up vector) keeps the horizon level: the camera's up axis
        always stays in the world-Z plane, so fly mode never rolls and the orbit controls behave
        exactly as before once fly mode is released.
        """
        forward = lookat - eye
        forward = forward / (np.linalg.norm(forward) + 1e-9)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right = right / (np.linalg.norm(right) + 1e-9)
        up = np.cross(right, forward)
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward  # pyrender cameras look down their local -Z
        pose[:3, 3] = eye
        return pose

    def _push() -> None:
        eye, lookat = controller.pose()
        gs_viewer.set_camera_pose(pose=_upright_pose(eye, lookat))
        # ``set_camera_pose`` only moves the camera; the trackball's orbit pivot (_target) is left
        # where it was, so a left-drag *after* flying would spin around a now-distant point. Re-anchor
        # the pivot a few metres ahead of the new camera so orbit resumes naturally around the view.
        trackball = getattr(pyrender_viewer, "_trackball", None)
        if trackball is not None:
            forward = lookat - eye
            forward = forward / (np.linalg.norm(forward) + 1e-9)
            pivot = eye + forward * 3.0
            trackball._target = pivot  # pyright: ignore[reportPrivateUsage]
            trackball._n_target = pivot  # pyright: ignore[reportPrivateUsage]

    class _FlyHandle:
        controller = None  # set below

        def engage(self) -> None:
            pose = getattr(pyrender_viewer._trackball, "pose", None)  # pyright: ignore[reportPrivateUsage]
            if pose is not None:
                eye = pose[:3, 3]
                controller.reseat(eye, eye + (-pose[:3, 2]))  # camera looks down local -z
            controller.set_active(True)

        def look(self, dx: float, dy: float) -> None:
            controller.on_look(dx, dy)
            _push()

    handle = _FlyHandle()
    handle.controller = controller  # type: ignore[assignment]
    pyrender_viewer._genelab_fly = handle  # pyright: ignore[reportAttributeAccessIssue]

    # Per-key wall-clock timestamp: HOLD callbacks fire at the (variable, often slow) sim-step
    # rate, so moving a fixed amount per tick is choppy and speed-inconsistent. Scaling each move
    # by the real elapsed time since that key last fired gives a steady ``move_speed`` m/s feel
    # regardless of frame rate. The first tick has no baseline (dt 0); dt is clamped so a stall
    # (e.g. window unfocused) can't teleport the camera on resume.
    last_fired: dict[str, float] = {}

    def _mover(action: str):
        def _cb() -> None:
            now = time.monotonic()
            dt = now - last_fired.get(action, now)
            last_fired[action] = now
            controller.move(action, min(dt, 0.1))
            if controller.active:
                _push()

        return _cb

    binds = [
        ("genelab_fly_forward", Key.W, "forward"),
        ("genelab_fly_back", Key.S, "back"),
        ("genelab_fly_left", Key.A, "left"),
        ("genelab_fly_right", Key.D, "right"),
        ("genelab_fly_up", Key.SPACE, "up"),
        ("genelab_fly_down", Key.LSHIFT, "down"),
        ("genelab_fly_down_r", Key.RSHIFT, "down"),
    ]
    # Never let an unexpected keybind conflict (or any Genesis-version drift in the keybind /
    # camera API) take down the whole viewer build — the fly camera is a convenience overlay.
    try:
        gs_viewer.register_keybinds(
            *[
                Keybind(name, key, key_action=KeyAction.HOLD, callback=_mover(action))
                for name, key, action in binds
            ],
            overwrite=True,
        )
    except Exception:
        _logger.warning(
            "fly camera: keybind registration failed; WASD navigation disabled", exc_info=True
        )
        pyrender_viewer._genelab_fly = None  # pyright: ignore[reportAttributeAccessIssue]


def _resolve_use_gpu(gpu: bool | None, device: str) -> bool:
    """Resolve the Genesis backend choice.

    ``gpu`` ``True`` / ``False`` force GPU / CPU; ``None`` follows ``device`` —
    ``cuda*`` runs the sim on the GPU, anything else on CPU. This keeps a
    ``device="cuda"`` env on the GPU backend without a separate opt-in.
    """
    if gpu is not None:
        return gpu
    return "cuda" in str(device).lower()


class InteractiveScene:
    """Owns the Genesis scene and the entity wrappers attached to it."""

    def __init__(
        self,
        sim_cfg: Any,
        scene_cfg: Any,
        *,
        device_hint: str,
    ) -> None:
        self._sim_cfg = sim_cfg
        self._scene_cfg = scene_cfg
        self._device = device_hint
        self._num_envs = max(1, int(sim_cfg.num_envs))
        self._gs_scene: Any = None
        self._env_origins: torch.Tensor = torch.zeros(self._num_envs, 3, device=device_hint)
        self._built = False
        # Set to True the first time ``gs_scene.step`` raises Genesis's
        # ``GenesisException("Viewer closed.")`` — i.e. the user closed the viewer
        # mid-rollout. Subsequent ``step`` calls become no-ops; consumers should
        # poll ``viewer_closed`` (via ``ManagerBasedRlEnv.viewer_closed``) to break
        # out of their loop cleanly.
        self._viewer_closed: bool = False
        self._gs_exception_cls: type[Exception] | None = None
        # Pre-allocate entity wrappers from cfg; ``add_entity`` may add more before ``build``.
        self._entities: dict[str, Articulation | RigidObject] = {}
        for name, entity_cfg in dict(scene_cfg.entities).items():
            self._entities[name] = self._make_entity(name, entity_cfg)
        # Terrain is optional. When None, ``build()`` adds a default flat plane (matches
        # M2 behavior); when set, the importer spawns a Genesis ``Terrain`` morph instead.
        self._terrain: TerrainImporter | None = (
            TerrainImporter(scene_cfg.terrain) if scene_cfg.terrain is not None else None
        )
        # Sensors are pre-built before ``gs_scene.build`` so resources that the renderer
        # snapshots at build time (e.g. cameras for BatchRenderer) are registered in
        # time. ``ManagerBasedRlEnv`` reuses these instances rather than re-building.
        self._sensors: dict[str, Sensor[Any]] = {}
        # Recorder bridge: only allocated when the scene cfg declares recordings. The
        # bridge holds late-bound env / sensor references used by data callables
        # registered with Genesis before ``gs_scene.build``.
        self._recorder_bridge: RecorderBridge | None = (
            RecorderBridge(scene=self)
            if tuple(getattr(scene_cfg, "recordings", ()) or ())
            else None
        )

    def add_entity(self, name: str, cfg: ArticulationCfg | RigidObjectCfg) -> None:
        """Register an additional entity before ``build()``."""
        if self._built:
            raise RuntimeError("InteractiveScene.add_entity called after build()")
        if name in self._entities:
            raise ValueError(f"duplicate entity name {name!r}")
        self._entities[name] = self._make_entity(name, cfg)

    @staticmethod
    def _make_entity(
        name: str, cfg: ArticulationCfg | RigidObjectCfg
    ) -> Articulation | RigidObject:
        if isinstance(cfg, ArticulationCfg):
            return Articulation(cfg, name=name)
        # ``cfg`` is RigidObjectCfg by the union; defensive runtime check for downstream
        # extensions that pass an unsupported type.
        if isinstance(cfg, RigidObjectCfg):  # pyright: ignore[reportUnnecessaryIsInstance]
            return RigidObject(cfg, name=name)
        raise TypeError(f"unknown entity cfg type for {name!r}: {type(cfg).__name__}")

    # ------------------------------------------------------------------ build

    def build(self) -> None:
        """Initialize Genesis, spawn every entity, build the parallel scene."""
        from genelab.utils.distributed import pin_cuda_device

        pinned = pin_cuda_device()
        if pinned is not None:
            # Distributed: force GPU backend and align device with rsl_rl's expected
            # ``cuda:{LOCAL_RANK}`` string. The CLI bootstrap has already set
            # CUDA_VISIBLE_DEVICES per rank and rewritten LOCAL_RANK to 0.
            self._sim_cfg.gpu = True
            self._device = pinned

        import genesis as gs  # type: ignore[import-not-found]

        # Cache the exception class so ``step`` can catch a viewer-closed event
        # without repeating the lazy import on the hot path.
        self._gs_exception_cls = gs.GenesisException

        gs_init = getattr(gs, "init", None)
        if gs_init is not None and not getattr(gs, "_initialized", False):
            use_gpu = _resolve_use_gpu(self._sim_cfg.gpu, self._device)
            backend = gs.gpu if use_gpu else gs.cpu  # type: ignore[attr-defined]
            gs.init(backend=backend, logging_level="warning")

        sim_options = gs.options.SimOptions(
            dt=float(self._sim_cfg.dt),
            substeps=int(self._sim_cfg.substeps),
        )
        # Optional rigid-solver tuning. The cfg keeps ``integrator`` as a
        # string (so ``configs`` stays Genesis-free); resolve it to ``gs.integrator.<name>``
        # here. ``rigid_options`` is only passed when the user set at least one field, so an
        # untouched config leaves Genesis's defaults exactly as before.
        rigid_kwargs = self._sim_cfg.rigid_options_kwargs()
        if "integrator" in rigid_kwargs:
            name = rigid_kwargs["integrator"]
            integrator = getattr(gs.integrator, name, None)
            if integrator is None:
                raise ValueError(f"SimulationCfg.integrator={name!r} is not a gs.integrator member")
            rigid_kwargs["integrator"] = integrator
        # ``batch_render=True`` swaps in Genesis's BatchRenderer so attached cameras can
        # emit per-env RGB-D tensors. Default ``None`` keeps the historic Rasterizer path.
        # ``use_rasterizer=False`` (default) keeps the raytracer; ``True`` switches to
        # the rasterizer for faster batched rendering.
        renderer = (
            gs.renderers.BatchRenderer(
                use_rasterizer=bool(getattr(self._scene_cfg, "use_rasterizer", False))
            )
            if getattr(self._scene_cfg, "batch_render", False)
            else None
        )
        # Render rate is configured separately from the physics rate: Genesis throttles
        # ``_visualizer.update`` via ``ViewerOptions.max_FPS``. We only construct
        # ``ViewerOptions`` when the viewer is enabled so headless training stays
        # untouched.
        # A non-empty ``panels`` list implies the ImGui overlay even if ``viewer_imgui`` was
        # left False, so "add a panel" is a one-liner that just works.
        enable_gui = bool(self._sim_cfg.viewer_imgui) or bool(
            getattr(self._sim_cfg, "panels", None)
        )
        viewer_options = (
            gs.options.ViewerOptions(**_viewer_option_kwargs(self._sim_cfg, enable_gui=enable_gui))
            if self._sim_cfg.vis
            else None
        )
        if self._sim_cfg.vis:
            # Workaround for an upstream pyrender bug that saves videos with a .png
            # extension. Applied once per process; no-op when not built.
            _patch_pyrender_save_filename()
            # Rebrand the viewer window caption from "Genesis <ver>" to "GeneLab <ver>".
            _patch_viewer_window_title()
            if enable_gui:
                # Workaround for Genesis/imgui_bundle overlay bugs (close crash, reversed scroll).
                _patch_imgui_overlay()
            if self._sim_cfg.fly_camera:
                # Right-mouse free-look (driven post-build by _install_fly_camera).
                _patch_viewer_fly_mouse()
        scene_kwargs: dict[str, Any] = dict(
            sim_options=sim_options,
            renderer=renderer,
            show_viewer=self._sim_cfg.vis,
        )
        if rigid_kwargs:
            scene_kwargs["rigid_options"] = gs.options.RigidOptions(**rigid_kwargs)
        if viewer_options is not None:
            scene_kwargs["viewer_options"] = viewer_options
        self._add_solver_options(gs, scene_kwargs)
        self._gs_scene = gs.Scene(**scene_kwargs)
        if self._terrain is None:
            self._gs_scene.add_entity(
                gs.morphs.Plane(
                    pos=(0.0, 0.0, self._scene_cfg.ground_plane_height),
                    collision=self._scene_cfg.ground_plane_collision,
                )
            )
        else:
            self._terrain.spawn(self._gs_scene)
        for entity in self._entities.values():
            entity.spawn(self._gs_scene)

        if self._sim_cfg.vis and self._scene_cfg.mouse_interaction:
            from genesis.vis.viewer_plugins import MouseInteractionPlugin

            # MouseInteractionPlugin must be attached BEFORE ``scene.build()`` — pre-build
            # registration routes through ``viewer.build`` so the plugin's raycaster sees the
            # fully constructed rigid solver. Post-build registration deadlocks the sim/viewer
            # loop in some Genesis builds.
            self._gs_scene.viewer.add_plugin(MouseInteractionPlugin(use_force=True))

        # Forward any custom ImGui panels to the overlay. ``register_panel`` only appends to a
        # copy-on-write list (drawn post-build on the viewer thread), so pre-build registration
        # is safe and keeps the panel host attached before ``build`` snapshots the viewer.
        if self._sim_cfg.vis and getattr(self._sim_cfg, "panels", None):
            register_viewer_panels(self._gs_scene.viewer, self._sim_cfg.panels, logger=_logger)

        # Pre-build sensors: instantiate every sensor and let it register any Genesis
        # resources (e.g. BatchRenderer cameras) before ``gs_scene.build`` snapshots
        # the camera list.
        sensor_cfgs = tuple(getattr(self._scene_cfg, "sensors", ()) or ())
        for sensor_cfg in sensor_cfgs:
            sensor = sensor_cfg.build()
            sensor.pre_build_genesis(self._gs_scene, dict(self._entities))
            self._sensors[sensor_cfg.name] = sensor

        # Register Genesis recorders before ``gs_scene.build`` (which calls
        # ``recorder_manager.build`` internally and flips it to ``assert_built``).
        if self._recorder_bridge is not None:
            from genelab.recording.register import register_recorders

            self._recorder_bridge.sensors = self._sensors
            self._recorder_bridge.entities = dict(self._entities)
            register_recorders(
                gs_scene=self._gs_scene,
                bridge=self._recorder_bridge,
                recording_cfgs=tuple(self._scene_cfg.recordings),
                physics_dt=float(self._sim_cfg.dt),
            )

        # A height-field terrain is a single large mesh shared by every env, with per-env
        # spawn origins spread across it (``TerrainImporter.spawn_pos`` / the terrain
        # curriculum). Genesis's ``env_spacing`` grid-offsets — and replicates — the whole
        # scene per env, so applying it on top of the terrain renders N copies of the
        # terrain shifted by the (small) spacing, heavily overlapping, with robots buried in
        # the neighbouring copies. Collapse the grid to zero when a terrain owns env layout;
        # flat-ground scenes keep the configured spacing to fan their robots out.
        env_spacing = (
            (0.0, 0.0) if self._terrain is not None else tuple(self._scene_cfg.env_spacing)
        )
        self._gs_scene.build(
            n_envs=self._num_envs,
            env_spacing=env_spacing,
        )

        self._device = str(gs.device)
        self._env_origins = self._compute_env_origins()

        # Post-build: bind every articulation so its introspection / per-joint tensors land.
        for entity in self._entities.values():
            if isinstance(entity, Articulation):
                entity.bind(self._num_envs, self._device)
        if self._terrain is not None:
            self._terrain.init_per_env_state(self._num_envs, self._device)
        if self._sim_cfg.vis and self._sim_cfg.fly_camera:
            # Viewer exists post-build: attach the right-mouse free-fly controller + WASD keybinds.
            _install_fly_camera(self._gs_scene.viewer, self._sim_cfg)
        self._built = True

    def _add_solver_options(self, gs: Any, scene_kwargs: dict[str, Any]) -> None:
        """Enable the deformable / fluid solvers this scene's materials require.

        Collects the union of ``required_solvers()`` across every entity material (plus
        any solver the user explicitly tuned via ``scene_cfg.solvers``) and sets the
        matching ``*_options`` kwarg on ``gs.Scene``. A solver the user tuned uses that
        cfg; one needed only because of a material gets Genesis defaults. Rigid /
        kinematic need no extra options, so a rigid-only scene leaves ``scene_kwargs``
        untouched — byte-for-byte identical to before materials existed.
        """
        from genelab.materials.options import SOLVER_SCENE_KWARGS

        required: set[str] = set()
        for entity in self._entities.values():
            material = getattr(entity.cfg, "material", None)
            if material is not None:
                required |= material.required_solvers()
        required &= set(SOLVER_SCENE_KWARGS)  # drop rigid / kinematic

        solvers_cfg = getattr(self._scene_cfg, "solvers", None)
        for family, (scene_kwarg, options_cls, attr) in SOLVER_SCENE_KWARGS.items():
            user_cfg = getattr(solvers_cfg, attr, None) if solvers_cfg is not None else None
            if user_cfg is None and family not in required:
                continue
            scene_kwargs[scene_kwarg] = (
                user_cfg.build(gs) if user_cfg is not None else getattr(gs.options, options_cls)()
            )

    def _compute_env_origins(self) -> torch.Tensor:
        scene_origins = getattr(self._gs_scene, "envs_offset", None)
        if scene_origins is None:
            scene_origins = getattr(self._gs_scene, "env_origins", None)
        if scene_origins is not None:
            return to_tensor(scene_origins, self._device).reshape(self._num_envs, 3)
        return torch.zeros(self._num_envs, 3, device=self._device)

    # ------------------------------------------------------------------ entity access

    def __getitem__(self, name: str) -> Articulation | RigidObject:
        return self._entities[name]

    def __contains__(self, name: object) -> bool:
        return name in self._entities

    def keys(self) -> Iterable[str]:
        return self._entities.keys()

    @property
    def articulations(self) -> dict[str, Articulation]:
        return {n: e for n, e in self._entities.items() if isinstance(e, Articulation)}

    @property
    def rigid_objects(self) -> dict[str, RigidObject]:
        return {n: e for n, e in self._entities.items() if isinstance(e, RigidObject)}

    # ------------------------------------------------------------------ runtime

    def step(self, *, update_visualizer: bool = True) -> None:
        """Step the Genesis scene by one physics tick.

        ``update_visualizer=False`` skips the viewer refresh (and its FPS-throttled
        ``rate.sleep``). ``ManagerBasedRlEnv`` uses this to render only on the last
        tick of its decimation loop so the viewer rate is decoupled from the physics
        rate. Callers that drive their own simulation loops can leave the default to
        keep the historic "render every tick" behavior.

        If the user closes the Genesis viewer mid-rollout, Genesis raises
        ``GenesisException("Viewer closed.")`` out of ``gs_scene.step``. The kernel
        catches that here, sets :py:attr:`viewer_closed`, and makes subsequent
        ``step`` calls no-ops — so every consumer (RL runner, showcase runner,
        bespoke scripts) only needs to poll the flag to exit cleanly instead of
        writing the same try / except / string-compare boilerplate. When inner
        physics ticks skip the viewer update, the close exception only fires on the
        next render tick, so closure is detected within one control step.
        """

        if self._viewer_closed:
            return
        exc_cls = self._gs_exception_cls
        try:
            self._gs_scene.step(update_visualizer=update_visualizer)
        except Exception as exc:
            if exc_cls is not None and isinstance(exc, exc_cls) and str(exc) == "Viewer closed.":
                self._viewer_closed = True
                return
            raise

    def refresh_state(self) -> None:
        for art in self.articulations.values():
            art.refresh()

    def reset(self, env_ids: torch.Tensor) -> None:
        for art in self.articulations.values():
            art.reset(env_ids)

    def draw_camera_frustums(
        self,
        *,
        camera_names: tuple[str, ...] | None = None,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.3),
    ) -> int:
        """Draw the view frustum of one or more :class:`~genelab.sensor.CameraSensor` s.

        Wraps Genesis 1.0's ``scene.draw_debug_frustum(camera, color=...)``. When
        ``camera_names`` is ``None`` every ``CameraSensor`` on the scene is drawn;
        otherwise only the named ones (a name that doesn't resolve to a
        ``CameraSensor`` raises). Returns the number of frustums drawn so callers
        can detect mis-named selections without inspecting the viewer state.

        Cameras allocate their Genesis-side handle in
        :meth:`~genelab.sensor.CameraSensor.pre_build_genesis`; calling this before
        :meth:`build` raises a ``RuntimeError``.
        """
        if not self._built:
            raise RuntimeError("draw_camera_frustums called before InteractiveScene.build()")
        selected = self._select_camera_sensors(camera_names)
        for sensor in selected:
            self._gs_scene.draw_debug_frustum(sensor.gs_camera, color=color)
        return len(selected)

    def draw_camera_trajectory(
        self,
        positions: Any,
        *,
        radius: float = 0.002,
        color: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.8),
    ) -> None:
        """Draw a polyline through ``positions`` (one row per world-frame point).

        Thin wrapper over Genesis 1.0's
        ``scene.draw_debug_trajectory(poss, radius=…, color=…)``. The caller owns
        the positions buffer (typically a recorded camera path or an inspection
        waypoint sequence) — no history is maintained on the scene side.
        """
        if not self._built:
            raise RuntimeError("draw_camera_trajectory called before InteractiveScene.build()")
        self._gs_scene.draw_debug_trajectory(positions, radius=radius, color=color)

    def _select_camera_sensors(self, names: tuple[str, ...] | None) -> "list[CameraSensor]":
        from genelab.sensor.camera import CameraSensor

        if names is None:
            return [s for s in self._sensors.values() if isinstance(s, CameraSensor)]
        selected: list[CameraSensor] = []
        for n in names:
            sensor = self._sensors.get(n)
            if sensor is None:
                raise KeyError(
                    f"draw_camera_frustums: no sensor named {n!r} on this scene "
                    f"(have: {sorted(self._sensors)})"
                )
            if not isinstance(sensor, CameraSensor):
                raise TypeError(
                    f"draw_camera_frustums: sensor {n!r} is not a CameraSensor "
                    f"(got {type(sensor).__name__})"
                )
            selected.append(sensor)
        return selected

    def close(self) -> None:
        scene = self._gs_scene
        if scene is None:
            return
        # Flush recorder buffers and join threads before destroying the Genesis scene.
        if self._recorder_bridge is not None:
            self._recorder_bridge.stop()
        for attr in ("close", "stop", "destroy"):
            fn = getattr(scene, attr, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                return

    # ------------------------------------------------------------------ properties

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return self._device

    @property
    def env_origins(self) -> torch.Tensor:
        return self._env_origins

    @property
    def gs_scene(self) -> Any:
        return self._gs_scene

    @property
    def terrain(self) -> TerrainImporter | None:
        """The active :class:`TerrainImporter`, or ``None`` for the default flat plane."""
        return self._terrain

    @property
    def viewer_closed(self) -> bool:
        """``True`` once the user closes the Genesis viewer mid-rollout.

        After the flag flips, :py:meth:`step` becomes a no-op so consumer loops can
        poll this property and break instead of catching
        ``GenesisException("Viewer closed.")`` themselves.
        """
        return self._viewer_closed

    @property
    def sensors(self) -> dict[str, Sensor[Any]]:
        """Sensors instantiated during ``build`` (keyed by ``SensorCfg.name``).

        Pre-build callers populate Genesis-side resources via
        :py:meth:`genelab.sensor.Sensor.pre_build_genesis`; ``ManagerBasedRlEnv``
        calls :py:meth:`bind` on each entry post-build to complete env-side wiring.
        """
        return self._sensors

    @property
    def recorder_bridge(self) -> RecorderBridge | None:
        """The recorder bridge if the scene cfg declared any ``recordings``, else ``None``."""
        return self._recorder_bridge


if TYPE_CHECKING:
    from typing import cast

    from genelab.contracts import SceneContext

    # InteractiveScene is the adapter for the SceneContext port. Type-only
    # assignment; pyright fails CI if the scene stops conforming.
    _scene_context_conformance: SceneContext = cast("InteractiveScene", ...)
