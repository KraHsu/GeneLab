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
    # Difficulty rank used only when ``TerrainGeneratorCfg.curriculum=True``: rows are
    # ordered easiest → hardest by this value (row 0 = lowest difficulty). Ignored for the
    # default proportion-weighted random tiling. Genesis keys ``subterrain_parameters`` by
    # type, so difficulty must vary across *distinct* sub-terrain types (in-type scaling
    # like "steeper stairs per row" is not expressible).
    difficulty: float = 0.0

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


@dataclass
class DiscreteObstaclesCfg(SubTerrainCfg):
    """Randomly placed rectangular obstacles (Genesis ``discrete_obstacles_terrain``).

    ``max_height`` is the obstacle height — Genesis draws each from
    ``±{max, max/2}``. ``min_size`` / ``max_size`` bound the rectangle side length (m);
    ``num_rects`` is the obstacle count. A useful "step over / around clutter" curriculum.
    """

    max_height: float = 0.05
    min_size: float = 1.0
    max_size: float = 2.0
    num_rects: int = 20

    def genesis_type(self) -> str:
        return "discrete_obstacles_terrain"

    def to_genesis_params(self) -> dict[str, Any]:
        return {
            "max_height": self.max_height,
            "min_size": self.min_size,
            "max_size": self.max_size,
            "num_rects": self.num_rects,
        }


@dataclass
class SteppingStonesCfg(SubTerrainCfg):
    """Grid of raised stones separated by gaps (Genesis ``stepping_stones_terrain``).

    ``stone_size`` is each stone's side length (m); ``stone_distance`` the gap between
    stones (m); ``max_height`` randomizes per-stone height; ``platform_size`` is the flat
    centre platform (m, ``0`` for none). Tests precise foot placement.
    """

    stone_size: float = 1.0
    stone_distance: float = 0.25
    max_height: float = 0.2
    platform_size: float = 1.0

    def genesis_type(self) -> str:
        return "stepping_stones_terrain"

    def to_genesis_params(self) -> dict[str, Any]:
        return {
            "stone_size": self.stone_size,
            "stone_distance": self.stone_distance,
            "max_height": self.max_height,
            "platform_size": self.platform_size,
        }


@dataclass
class FractalCfg(SubTerrainCfg):
    """Multi-octave fractal-noise terrain (Genesis ``fractal_terrain``).

    ``levels`` is the number of octaves (more = finer detail); ``scale`` the overall
    amplitude. A natural-looking rough ground without the blocky steps of
    ``random_uniform_terrain``.
    """

    levels: int = 8
    scale: float = 5.0

    def genesis_type(self) -> str:
        return "fractal_terrain"

    def to_genesis_params(self) -> dict[str, Any]:
        return {"levels": self.levels, "scale": self.scale}
