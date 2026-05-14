"""Terrain importer: bridges GeneLab cfg to Genesis ``gs.morphs.Terrain``.

The importer is a thin spawn wrapper that mirrors :class:`genelab.entity.RigidObject`'s
two-phase API. :class:`genelab.scene.InteractiveScene` instantiates it pre-build, then
calls :meth:`spawn` to add the terrain morph to the Genesis scene. After ``scene.build``
returns, :attr:`heightfield` and :attr:`terrain_scale` expose the post-construction
data Genesis fills in (used by the ray-cast sensor in PR2).
"""

from typing import Any

import torch

from genelab.terrains.generator import TerrainGenerator, TerrainGeneratorCfg


class TerrainImporter:
    """Owns a :class:`TerrainGenerator` and the spawned Genesis terrain entity."""

    def __init__(self, cfg: TerrainGeneratorCfg) -> None:
        self.cfg = cfg
        self.generator = TerrainGenerator(cfg)
        self._gs_handle: Any = None

    # ------------------------------------------------------------------ lifecycle

    def spawn(self, gs_scene: Any) -> None:
        """Add a ``gs.morphs.Terrain`` to ``gs_scene``. Must be called pre-build."""
        import genesis as gs  # type: ignore[import-not-found]

        morph = gs.morphs.Terrain(**self.generator.genesis_kwargs)
        self._gs_handle = gs_scene.add_entity(morph)

    # ------------------------------------------------------------------ post-spawn data

    @property
    def heightfield(self) -> Any:
        """Raw integer height field, shape ``(H, W)``.

        Multiply by ``terrain_scale[1]`` (vertical scale) to recover meters.
        """
        if self._gs_handle is None:
            raise RuntimeError("TerrainImporter.heightfield accessed before spawn()")
        return self._gs_handle.terrain_hf

    @property
    def terrain_scale(self) -> Any:
        """``(horizontal_scale, vertical_scale)`` reported by Genesis post-spawn."""
        if self._gs_handle is None:
            raise RuntimeError("TerrainImporter.terrain_scale accessed before spawn()")
        return self._gs_handle.terrain_scale

    # ------------------------------------------------------------------ pass-throughs

    @property
    def env_origins(self) -> torch.Tensor:
        return self.generator.env_origins

    @property
    def layout(self) -> list[list[str]]:
        return self.generator.layout

    @property
    def gs_handle(self) -> Any:
        return self._gs_handle
