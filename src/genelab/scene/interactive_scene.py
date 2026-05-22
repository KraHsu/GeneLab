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

import os
from collections.abc import Iterable
from typing import Any

import torch

from genelab.entity import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from genelab.entity._torch import to_tensor
from genelab.recording.bridge import RecorderBridge
from genelab.sensor import Sensor
from genelab.terrains import TerrainImporter


_pyrender_save_patch_applied = False


def _patch_pyrender_save_filename() -> None:
    """Coerce pyrender's tkinter SaveAs default extension to match the requested one.

    Upstream pyrender hard-codes ``defaultextension=".png"`` in ``_get_save_filename``,
    so pressing ``R`` twice to save a recorded video (called with ``file_exts=["mp4"]``)
    and typing a bare filename produces e.g. ``dancing.png`` — and ``Viewer.save_video``
    then ``shutil.move``s the .mp4 file to that .png path. We can't modify Genesis, so
    we wrap the method at the class level: when exactly one extension is requested and
    the dialog returned a different one, swap the extension. Multi-extension dialogs
    (e.g. ``_save_image`` offers png/jpg/gif/all) are left alone so the user's chosen
    type is respected.
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
            backend = gs.gpu if self._sim_cfg.gpu else gs.cpu  # type: ignore[attr-defined]
            gs.init(backend=backend, logging_level="warning")

        sim_options = gs.options.SimOptions(
            dt=float(self._sim_cfg.dt),
            substeps=int(self._sim_cfg.substeps),
        )
        # ``batch_render=True`` swaps in Genesis's BatchRenderer so attached cameras can
        # emit per-env RGB-D tensors. Default ``None`` keeps the historic Rasterizer path.
        renderer = (
            gs.renderers.BatchRenderer(use_rasterizer=False)
            if getattr(self._scene_cfg, "batch_render", False)
            else None
        )
        # Render rate is configured separately from the physics rate: Genesis throttles
        # ``_visualizer.update`` via ``ViewerOptions.max_FPS``. We only construct
        # ``ViewerOptions`` when the viewer is enabled so headless training stays
        # untouched.
        viewer_options = (
            gs.options.ViewerOptions(max_FPS=self._sim_cfg.render_fps)
            if self._sim_cfg.vis
            else None
        )
        if self._sim_cfg.vis:
            # Workaround for an upstream pyrender bug that saves videos with a .png
            # extension. Applied once per process; no-op when not built.
            _patch_pyrender_save_filename()
        scene_kwargs: dict[str, Any] = dict(
            sim_options=sim_options,
            renderer=renderer,
            show_viewer=self._sim_cfg.vis,
        )
        if viewer_options is not None:
            scene_kwargs["viewer_options"] = viewer_options
        self._gs_scene = gs.Scene(**scene_kwargs)
        if self._terrain is None:
            self._gs_scene.add_entity(gs.morphs.Plane())
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

        self._gs_scene.build(
            n_envs=self._num_envs,
            env_spacing=tuple(self._scene_cfg.env_spacing),
        )

        gs_device = getattr(gs, "device", None)
        if gs_device is not None:
            try:
                self._device = str(gs_device())  # type: ignore[misc]
            except TypeError:
                self._device = str(gs_device)
        self._env_origins = self._compute_env_origins()

        # Post-build: bind every articulation so its introspection / per-joint tensors land.
        for entity in self._entities.values():
            if isinstance(entity, Articulation):
                entity.bind(self._num_envs, self._device)
        if self._terrain is not None:
            self._terrain.init_per_env_state(self._num_envs, self._device)
        self._built = True

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
