"""Spatial footprint memory map for deformable terrain (ADR-0001 Q3, paper §6.5).

A per-environment 2D grid that stores plastic residual keyed by *world position*. Unlike
a per-foot residual scalar, the map couples feet spatially: a footprint stamped by one
foot is read back by any foot that later visits the same cell ("front foot → rear foot",
"don't step back into the hole you dug").
"""

from __future__ import annotations

import torch


class FootprintMap:
    """Per-env world-keyed grid of plastic residual.

    The grid covers ``[origin - size/2, origin + size/2]`` in x and y at ``resolution``
    metres per cell. Positions outside the grid clamp to the nearest edge cell.
    """

    def __init__(
        self,
        num_envs: int,
        resolution: float,
        size: tuple[float, float],
        origin: tuple[float, float] = (0.0, 0.0),
        device: torch.device | str = "cpu",
    ) -> None:
        self.resolution = resolution
        self.size = size
        self.origin = origin
        self.nx = max(1, round(size[0] / resolution))
        self.ny = max(1, round(size[1] / resolution))
        self.grid = torch.zeros(num_envs, self.nx * self.ny, device=device)

    def _cell_index(self, pos_xy: torch.Tensor) -> torch.Tensor:
        """Map world ``(x, y)`` positions to flattened cell indices, shape ``(E, P)``."""
        x = (pos_xy[..., 0] - self.origin[0] + self.size[0] / 2.0) / self.resolution
        y = (pos_xy[..., 1] - self.origin[1] + self.size[1] / 2.0) / self.resolution
        ix = torch.floor(x).long().clamp(0, self.nx - 1)
        iy = torch.floor(y).long().clamp(0, self.ny - 1)
        return ix * self.ny + iy

    def read(self, pos_xy: torch.Tensor) -> torch.Tensor:
        """Residual at each ``(x, y)`` position, shape ``(num_envs, num_points)``."""
        return torch.gather(self.grid, 1, self._cell_index(pos_xy))

    def accumulate(self, pos_xy: torch.Tensor, amount: torch.Tensor) -> None:
        """Add ``amount`` to the cell under each ``(x, y)`` position."""
        self.grid.scatter_add_(1, self._cell_index(pos_xy), amount)

    def recover(self, dt: float, recovery_time: float) -> None:
        """Relax the whole map toward zero one step: ``M -= (dt / recovery_time) * M``."""
        self.grid -= (dt / recovery_time) * self.grid

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear the map for ``env_ids`` (or all environments when ``None``)."""
        if env_ids is None:
            self.grid.zero_()
        else:
            self.grid[env_ids] = 0.0
