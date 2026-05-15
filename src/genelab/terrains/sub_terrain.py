"""Sub-terrain dataclass configs.

Each cfg maps onto one of Genesis's built-in subterrain string types and surfaces the
per-type parameters as plain dataclass fields. The translation back to Genesis kwargs
happens in :class:`genelab.terrains.generator.TerrainGenerator`.

The :class:`SubTerrainCfg` base is abstract: subclasses override ``genesis_type`` and
``to_genesis_params``. ``proportion`` controls random tiling when
``TerrainGeneratorCfg.layout`` is left at ``None``.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class SubTerrainCfg:
    """Base config for a single sub-terrain cell type."""

    proportion: float = 1.0

    def genesis_type(self) -> str:
        raise NotImplementedError("SubTerrainCfg subclasses must override genesis_type")

    def to_genesis_params(self) -> dict[str, Any]:
        return {}


@dataclass
class FlatPatchCfg(SubTerrainCfg):
    """A flat patch at z=0. Useful as the easy end of a difficulty curriculum."""

    def genesis_type(self) -> str:
        return "flat_terrain"


@dataclass
class PyramidStairsCfg(SubTerrainCfg):
    """Concentric square stairs descending from the patch center.

    ``step_height`` is negative for descending stairs (matching Genesis defaults). The
    flat plateau at the center is always one ``step_width`` wide.
    """

    step_width: float = 0.75
    step_height: float = -0.1

    def genesis_type(self) -> str:
        return "pyramid_stairs_terrain"

    def to_genesis_params(self) -> dict[str, Any]:
        return {"step_width": self.step_width, "step_height": self.step_height}


@dataclass
class RandomRoughCfg(SubTerrainCfg):
    """Uniformly random small bumps. Bumps are sampled at ``downsampled_scale`` then
    upsampled to the patch resolution, so the apparent feature size is independent of
    ``horizontal_scale``."""

    min_height: float = -0.1
    max_height: float = 0.1
    step: float = 0.1
    downsampled_scale: float = 0.5

    def genesis_type(self) -> str:
        return "random_uniform_terrain"

    def to_genesis_params(self) -> dict[str, Any]:
        return {
            "min_height": self.min_height,
            "max_height": self.max_height,
            "step": self.step,
            "downsampled_scale": self.downsampled_scale,
        }


@dataclass
class SlopeCfg(SubTerrainCfg):
    """Linearly inclined patch.

    ``slope`` is signed: negative tilts down along Genesis's default direction,
    matching the ``-0.5`` default that ``gs.morphs.Terrain`` ships for the
    built-in ``sloped_terrain`` preset.
    """

    slope: float = -0.5

    def genesis_type(self) -> str:
        return "sloped_terrain"

    def to_genesis_params(self) -> dict[str, Any]:
        return {"slope": self.slope}


@dataclass
class WaveCfg(SubTerrainCfg):
    """Sinusoidal undulations across the patch.

    ``num_waves`` is the wave-cycle count across the cell; ``amplitude`` is the
    half peak-to-peak height in metres. Defaults match Genesis built-ins
    (``num_waves=2.0``, ``amplitude=0.1``).
    """

    num_waves: float = 2.0
    amplitude: float = 0.1

    def genesis_type(self) -> str:
        return "wave_terrain"

    def to_genesis_params(self) -> dict[str, Any]:
        return {"num_waves": self.num_waves, "amplitude": self.amplitude}
