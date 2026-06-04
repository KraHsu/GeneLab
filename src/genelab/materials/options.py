"""Solver-option wrappers and the scene-facing solver group.

A deformable / fluid material is inert unless ``gs.Scene`` is built with the matching
solver options (``mpm_options``, ``fem_options``, …). :class:`InteractiveScene`
auto-enables a solver with Genesis defaults whenever an entity carries a material of
that family; supply the matching ``*OptionsCfg`` here through
:class:`SolverOptionsCfg` to tune the domain bounds / iteration counts instead.

Like the material cfgs, every field defaults to ``None`` ("use the Genesis default")
and :meth:`build` forwards only the set fields. The module imports no ``genesis`` at
load time.
"""

from dataclasses import dataclass, fields
from typing import Any, ClassVar, Literal


@dataclass
class _SolverOptionsCfg:
    """Base for ``gs.options.*Options`` wrappers."""

    _gs_options: ClassVar[str] = ""

    def _kwargs(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                out[f.name] = value
        return out

    def build(self, gs: Any) -> Any:
        return getattr(gs.options, type(self)._gs_options)(**self._kwargs())


@dataclass
class MpmOptionsCfg(_SolverOptionsCfg):
    """``gs.options.MPMOptions`` — MPM domain bounds and grid resolution."""

    _gs_options: ClassVar[str] = "MPMOptions"
    dt: float | None = None
    gravity: tuple[float, float, float] | None = None
    particle_size: float | None = None
    grid_density: float | None = None
    lower_bound: tuple[float, float, float] | None = None
    upper_bound: tuple[float, float, float] | None = None


@dataclass
class FemOptionsCfg(_SolverOptionsCfg):
    """``gs.options.FEMOptions`` — FEM damping and Newton/PCG solver settings."""

    _gs_options: ClassVar[str] = "FEMOptions"
    dt: float | None = None
    gravity: tuple[float, float, float] | None = None
    damping: float | None = None
    floor_height: float | None = None
    use_implicit_solver: bool | None = None
    n_newton_iterations: int | None = None
    n_pcg_iterations: int | None = None


@dataclass
class PbdOptionsCfg(_SolverOptionsCfg):
    """``gs.options.PBDOptions`` — PBD iteration counts and domain bounds."""

    _gs_options: ClassVar[str] = "PBDOptions"
    dt: float | None = None
    gravity: tuple[float, float, float] | None = None
    particle_size: float | None = None
    max_stretch_solver_iterations: int | None = None
    max_bending_solver_iterations: int | None = None
    max_volume_solver_iterations: int | None = None
    max_density_solver_iterations: int | None = None
    max_viscosity_solver_iterations: int | None = None
    lower_bound: tuple[float, float, float] | None = None
    upper_bound: tuple[float, float, float] | None = None


@dataclass
class SphOptionsCfg(_SolverOptionsCfg):
    """``gs.options.SPHOptions`` — SPH pressure solver and domain bounds."""

    _gs_options: ClassVar[str] = "SPHOptions"
    dt: float | None = None
    gravity: tuple[float, float, float] | None = None
    particle_size: float | None = None
    pressure_solver: Literal["WCSPH", "DFSPH"] | None = None
    lower_bound: tuple[float, float, float] | None = None
    upper_bound: tuple[float, float, float] | None = None


@dataclass
class SfOptionsCfg(_SolverOptionsCfg):
    """``gs.options.SFOptions`` — stable-fluid grid resolution and inlet."""

    _gs_options: ClassVar[str] = "SFOptions"
    dt: float | None = None
    res: int | None = None
    solver_iters: int | None = None
    decay: float | None = None
    inlet_pos: tuple[float, float, float] | None = None
    inlet_vel: tuple[float, float, float] | None = None


@dataclass
class ToolOptionsCfg(_SolverOptionsCfg):
    """``gs.options.ToolOptions`` — tool-solver timestep / floor."""

    _gs_options: ClassVar[str] = "ToolOptions"
    dt: float | None = None
    floor_height: float | None = None


@dataclass
class SolverOptionsCfg:
    """Per-solver overrides for non-rigid materials, read by the scene.

    Each field is ``None`` until tuned. The scene maps a material's solver family to
    the matching field here; when a field is set, its options replace the Genesis
    defaults the scene would otherwise pass.
    """

    mpm: MpmOptionsCfg | None = None
    fem: FemOptionsCfg | None = None
    pbd: PbdOptionsCfg | None = None
    sph: SphOptionsCfg | None = None
    sf: SfOptionsCfg | None = None
    tool: ToolOptionsCfg | None = None


# Solver family -> (gs.Scene kwarg, gs.options class, SolverOptionsCfg field). The
# scene uses this to enable exactly the solvers its entities' materials require.
SOLVER_SCENE_KWARGS: dict[str, tuple[str, str, str]] = {
    "mpm": ("mpm_options", "MPMOptions", "mpm"),
    "fem": ("fem_options", "FEMOptions", "fem"),
    "pbd": ("pbd_options", "PBDOptions", "pbd"),
    "sph": ("sph_options", "SPHOptions", "sph"),
    "sf": ("sf_options", "SFOptions", "sf"),
    "tool": ("tool_options", "ToolOptions", "tool"),
}
