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

from collections.abc import Iterable
from typing import Any

import torch

from genelab.entity import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from genelab.entity._torch import to_tensor
from genelab.terrains import TerrainImporter


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
        # Pre-allocate entity wrappers from cfg; ``add_entity`` may add more before ``build``.
        self._entities: dict[str, Articulation | RigidObject] = {}
        for name, entity_cfg in dict(scene_cfg.entities).items():
            self._entities[name] = self._make_entity(name, entity_cfg)
        # Terrain is optional. When None, ``build()`` adds a default flat plane (matches
        # M2 behavior); when set, the importer spawns a Genesis ``Terrain`` morph instead.
        self._terrain: TerrainImporter | None = (
            TerrainImporter(scene_cfg.terrain) if scene_cfg.terrain is not None else None
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
        from genelab.rl.distributed import pin_cuda_device

        pinned = pin_cuda_device()
        if pinned is not None:
            # Distributed: force GPU backend and align device with rsl_rl's expected
            # ``cuda:{LOCAL_RANK}`` string. The CLI bootstrap has already set
            # CUDA_VISIBLE_DEVICES per rank and rewritten LOCAL_RANK to 0.
            self._sim_cfg.gpu = True
            self._device = pinned

        import genesis as gs  # type: ignore[import-not-found]

        gs_init = getattr(gs, "init", None)
        if gs_init is not None and not getattr(gs, "_initialized", False):
            backend = gs.gpu if self._sim_cfg.gpu else gs.cpu  # type: ignore[attr-defined]
            gs.init(backend=backend, logging_level="warning")

        sim_options = gs.options.SimOptions(
            dt=float(self._sim_cfg.dt),
            substeps=int(self._sim_cfg.substeps),
        )
        self._gs_scene = gs.Scene(
            sim_options=sim_options,
            show_viewer=self._sim_cfg.vis,
        )
        if self._terrain is None:
            self._gs_scene.add_entity(gs.morphs.Plane())
        else:
            self._terrain.spawn(self._gs_scene)
        for entity in self._entities.values():
            entity.spawn(self._gs_scene)

        if self._sim_cfg.vis and self._scene_cfg.mouse_interaction:
            from genelab.viewer.mouse_interaction import GeneLabMouseInteractionPlugin

            # MouseInteractionPlugin must be attached BEFORE ``scene.build()`` — pre-build
            # registration routes through ``viewer.build`` so the plugin's raycaster sees the
            # fully constructed rigid solver. Post-build registration deadlocks the sim/viewer
            # loop in some Genesis builds.
            self._gs_scene.viewer.add_plugin(GeneLabMouseInteractionPlugin(use_force=True))

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

    def step(self) -> None:
        self._gs_scene.step()

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
