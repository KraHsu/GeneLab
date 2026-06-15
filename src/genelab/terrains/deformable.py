"""Analytic deformable-terrain force laws (ADR-0001 stage 0).

The compliance-only degenerate branch of the stateful deformable terrain model. The
normal support law here is the Genesis-free core validated end-to-end against the
rigid solver in ``plans/spikes/analytic_sinkage_spike.py``: with the foot<->terrain
rigid contact suppressed, injecting this force at the foot link settles it at a soft
sinkage equilibrium ``d_eq = W / k``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

from genelab.terrains.footprint_map import FootprintMap


def compliant_normal_force(
    depth: torch.Tensor,
    depth_rate: torch.Tensor,
    k: float | torch.Tensor,
    c: float,
) -> torch.Tensor:
    """Unilateral compliant normal support force ``F_z = max(0, k*depth + c*depth_rate)``.

    Args:
        depth: Penetration of the foot below the undisturbed surface; ``> 0`` means
            the foot is sunk into the terrain, ``<= 0`` means it is at or above the
            surface (no contact).
        depth_rate: Rate of change of ``depth``; ``> 0`` means sinking deeper.
        k: Effective terrain stiffness (N/m).
        c: Effective terrain damping (N·s/m).

    Returns:
        Upward support force, never negative (the terrain pushes, never pulls).
    """
    in_contact = depth > 0.0
    force = k * depth + c * depth_rate
    return torch.where(in_contact, force, torch.zeros_like(force)).clamp_min(0.0)


NormalLaw: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor, "float | torch.Tensor", float], torch.Tensor
]
"""Signature of a pluggable normal support law: ``(depth, depth_rate, k, c) -> F_z``."""


def coulomb_tangential_force(
    normal_force: torch.Tensor,
    slip_velocity: torch.Tensor,
    mu: float | torch.Tensor,
    *,
    shear_max: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Regularized Coulomb friction opposing horizontal foot slip (paper §6.3).

    ``F_xy = -min(mu * F_z, shear_max) * v_xy / (|v_xy| + eps)``: opposes the slip
    direction with magnitude approaching the traction cap once the slip speed clears
    ``eps``, and smoothly vanishing as the foot stops (the regularization avoids the
    discontinuity of ideal Coulomb friction at zero slip).

    The cap is the Coulomb limit ``mu * F_z`` unless a granular shear strength
    ``shear_max`` is lower — on soft/sand terrain traction saturates at the terrain's
    shear strength rather than rising indefinitely with normal load.

    Args:
        normal_force: Upward support magnitude ``F_z >= 0``, shape ``(..., )``.
        slip_velocity: Horizontal foot velocity, shape ``(..., 2)``.
        mu: Effective friction coefficient.
        shear_max: Optional per-foot shear-strength ceiling (N), shape ``(..., )``.
        eps: Slip-speed regularization (m/s).

    Returns:
        Tangential force opposing slip, shape ``(..., 2)``; ``|F_xy| <= cap``.
    """
    speed = slip_velocity.norm(dim=-1, keepdim=True)
    cap = mu * normal_force
    if shear_max is not None:
        cap = torch.minimum(cap, shear_max)
    return -cap.unsqueeze(-1) * slip_velocity / (speed + eps)


def stick_slip_force(
    anchor_xy: torch.Tensor,
    foot_pos_xy: torch.Tensor,
    normal_force: torch.Tensor,
    *,
    k_lat: float,
    mu: float | torch.Tensor,
    shear_max: torch.Tensor | None = None,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchored (stick-slip) tangential force — static friction + sinkage shear hold.

    A planted foot is tied to an anchor point ``anchor_xy`` by a lateral spring:
    ``F_xy = -k_lat * (foot_pos_xy - anchor_xy)``, saturated at the Coulomb / shear cap
    ``min(mu*F_z, shear_max)``. Below the cap the foot *sticks* (a restoring force that
    resists lateral displacement even at zero velocity — the lateral hold a real planted /
    sunk foot gives, which the velocity-only Coulomb model lacks). At the cap it *slips*
    and the anchor drags so the force stays at the limit.

    Caller contract: set the anchor to the foot position on touchdown (a fresh, force-free
    anchor) and feed the returned ``new_anchor`` back each step while in contact.

    Args:
        anchor_xy: Current anchor (world x, y), shape ``(..., 2)``.
        foot_pos_xy: Current foot (world x, y), shape ``(..., 2)``.
        normal_force: Support magnitude ``F_z >= 0``, shape ``(...,)``.
        k_lat: Lateral (bristle) stiffness (N/m).
        mu: Friction coefficient (Coulomb cap ``mu * F_z``).
        shear_max: Optional granular shear-strength cap (N), shape ``(...,)``.
        eps: Numerical guard.

    Returns:
        ``(force, new_anchor)`` — lateral force ``(..., 2)`` and the updated anchor.
    """
    force = -k_lat * (foot_pos_xy - anchor_xy)
    mag = force.norm(dim=-1, keepdim=True)
    cap = (mu * normal_force).unsqueeze(-1)
    if shear_max is not None:
        cap = torch.minimum(cap, shear_max.unsqueeze(-1))
    scale = torch.where(mag > cap, cap / (mag + eps), torch.ones_like(mag))
    force = force * scale
    new_anchor = foot_pos_xy + force / k_lat
    return force, new_anchor


def granular_drag_force(
    depth: torch.Tensor,
    velocity_xy: torch.Tensor,
    drag_coeff: float | torch.Tensor,
    *,
    v_ref: float = 0.05,
) -> torch.Tensor:
    """Rate-independent granular plowing drag on a submerged foot.

    Sweeping a buried intruder horizontally through granular media mobilizes the
    passive failure of the soil wedge ahead of it: the resistance integrates the
    lithostatic pressure over the buried frontal area, giving ``F = K_d * depth^2``
    opposing the motion — *independent of the normal load* and (to first order in
    quasistatic regimes) of speed. This is the cost the velocity-Coulomb model misses:
    under a soft support (small ``F_z``), ``mu * F_z`` makes dragging a deeply buried
    foot nearly free, which lets a policy "paddle" through sand instead of stepping
    over it.

    Regularized near rest by ``|v| / (|v| + v_ref)`` so a planted foot feels no
    chattering drag (the static hold is :func:`stick_slip_force`'s job).

    Args:
        depth: Per-foot sinkage (m), shape ``(...,)``; ``<= 0`` (above the surface)
            contributes nothing.
        velocity_xy: Per-foot horizontal velocity, shape ``(..., 2)``.
        drag_coeff: Plowing gain ``K_d`` (N/m²) — roughly ``K_p * rho * g * w / 2``
            for soil density ``rho``, intruder width ``w``, passive coefficient ``K_p``.
        v_ref: Regularization velocity (m/s).

    Returns:
        Drag force per foot, shape ``(..., 2)``, opposing ``velocity_xy``.
    """
    magnitude = (drag_coeff * depth.clamp_min(0.0).square()).unsqueeze(-1)
    speed = velocity_xy.norm(dim=-1, keepdim=True)
    return -magnitude * velocity_xy / (speed + v_ref)


def granular_extraction_force(
    depth: torch.Tensor,
    foot_vel_z: torch.Tensor,
    extraction_coeff: float | torch.Tensor,
    *,
    v_ref: float = 0.05,
) -> torch.Tensor:
    """Overburden resistance against pulling a buried foot up out of granular media.

    Lifting a buried intruder mobilizes the weight and shaft friction of the material
    above it — ``F_z = -K_e * depth^2`` opposing the lift, the vertical sibling of
    :func:`granular_drag_force`. Without it the analytic surrogate prices extraction at
    zero, so a policy trained on it high-steps for free while the same motion is costly
    on real granular media — the two optima diverge. Descent is *not* resisted here:
    penetration is the compliance/damping model's regime.

    Regularized near rest by ``v_up / (v_up + v_ref)`` like the horizontal drag.

    Args:
        depth: Per-foot sinkage (m), shape ``(...,)``; ``<= 0`` contributes nothing.
        foot_vel_z: Per-foot vertical velocity (m/s); positive = lifting.
        extraction_coeff: Overburden gain ``K_e`` (N/m²).
        v_ref: Regularization velocity (m/s).

    Returns:
        Vertical force per foot, shape ``(...,)`` — ``<= 0``, resisting the lift.
    """
    magnitude = extraction_coeff * depth.clamp_min(0.0).square()
    v_up = foot_vel_z.clamp_min(0.0)
    return -magnitude * v_up / (v_up + v_ref)


def pneumatic_normal_force(
    depth: torch.Tensor,
    depth_rate: torch.Tensor,
    k: float | torch.Tensor,
    c: float,
    *,
    capacity: float,
    min_headroom: float = 0.01,
) -> torch.Tensor:
    """Sealed-air-chamber support (air mattress): one pressure, shared by all feet.

    The chamber's gauge pressure is an instantaneous function of the *total* volume the
    feet displace — isothermal ideal gas, ``P ∝ D / (capacity - D)`` with
    ``D = Σ max(depth_i, 0)`` summed over the feet axis. Every foot in contact feels the
    same pressure (cross-foot coupling: pressing one corner pushes the others up), feet
    above the surface feel nothing, and the force blows up as the chamber bottoms out.

    A :data:`NormalLaw` — bind ``capacity`` with ``functools.partial`` and set it as
    ``DeformableTerrainCfg.normal_law``. Per-foot damping ``c * depth_rate`` is kept for
    numerical stability, as in :func:`compliant_normal_force`.

    Args:
        depth: Per-foot sinkage, shape ``(num_envs, num_feet)``; the feet axis is the
            chamber (one chamber per env).
        depth_rate: Per-foot sinkage rate.
        k: Pressure stiffness (N/m of total displaced depth at small fill).
        c: Per-foot damping (N·s/m).
        capacity: Total displaced depth at which the chamber bottoms out (m).
        min_headroom: Fraction of ``capacity`` kept as headroom so the force stays
            finite at full compression.

    Returns:
        Per-foot support, shape like ``depth``; zero for feet above the surface.
    """
    displaced = depth.clamp_min(0.0).sum(dim=-1, keepdim=True)
    headroom = (1.0 - displaced / capacity).clamp_min(min_headroom)
    pressure_force = k * displaced / headroom
    in_contact = depth > 0.0
    force = pressure_force.expand_as(depth) + c * depth_rate
    return torch.where(in_contact, force, torch.zeros_like(force)).clamp_min(0.0)


def plastic_residual_update(
    residual: torch.Tensor,
    depth: torch.Tensor,
    dt: float,
    *,
    yield_depth: float,
    plastic_rate: float,
    recovery_time: float,
) -> torch.Tensor:
    """One step of plastic-residual evolution — the terrain's footprint memory (paper §6.4).

    ``r' = r + plastic_rate * max(0, depth - yield_depth) * dt - (dt / recovery_time) * r``

    When the foot presses past ``yield_depth`` the terrain takes a permanent set (``r``
    grows); between presses that set relaxes back toward zero with time constant
    ``recovery_time``. This is what makes the terrain *stateful*: a footprint left by one
    contact persists and influences later contacts.

    Args:
        residual: Current plastic residual per foot, shape ``(...,)``.
        depth: Current penetration below the surface, shape ``(...,)``.
        dt: Physics timestep (s).
        yield_depth: Penetration beyond which plastic deformation accumulates (m).
        plastic_rate: Accumulation gain (1/s) per metre of over-yield penetration.
        recovery_time: Relaxation time constant ``tau_rec`` (s); large = near-permanent.

    Returns:
        The residual after one step, same shape as ``residual``.
    """
    accumulation = plastic_rate * (depth - yield_depth).clamp_min(0.0) * dt
    recovery = (dt / recovery_time) * residual
    return residual + accumulation - recovery


@dataclass
class DeformableTerrainCfg:
    """Config for the compliance-only degenerate branch (ADR-0001 stage 0).

    The collision groups encode the Spike-B suppression mechanism: a soft-terrain geom
    whose ``contype``/``conaffinity`` bits are *disjoint* from the feet's bypasses the
    native rigid contact, so the analytic ``F_z`` is the foot's only support. Hard
    terrain keeps a group that *intersects* the feet's, so it collides normally — which
    is what makes hard↔soft transitions fall out for free.
    """

    k: float = 1.0e4
    """Effective terrain stiffness (N/m)."""
    c: float = 200.0
    """Effective terrain damping (N·s/m)."""
    normal_law: "NormalLaw" = compliant_normal_force
    """Normal support force law ``(depth, depth_rate, k, c) -> F_z``. The model calls it
    with the residual-corrected elastic depth and the per-env ``k`` tensor. Default is
    the linear unilateral spring-damper :func:`compliant_normal_force`; swap in a custom
    pure function for nonlinear media (pneumatic bottoming-out, membrane tension, ...) —
    law-specific constants go in a closure or ``functools.partial``."""
    mu: float = 0.0
    """Effective friction coefficient. ``0`` (default) = frictionless (stage-0 behavior)."""
    eta: float | None = None
    """Granular shear-strength ceiling on traction (N). ``None`` = pure Coulomb ``mu*F_z``."""
    lateral_stiffness: float = 0.0
    """Stick-slip bristle stiffness (N/m). ``> 0`` switches the tangential model from
    velocity-only Coulomb to anchored static friction + sinkage shear hold (a planted foot
    resists lateral displacement even at zero velocity). ``0`` (default) = kinetic Coulomb."""
    drag_coeff: float = 0.0
    """Granular plowing drag gain ``K_d`` (N/m²): sweeping a buried foot horizontally
    costs ``K_d * depth²`` regardless of normal load (see :func:`granular_drag_force`).
    ``0`` (default) = no plowing drag."""
    extraction_coeff: float = 0.0
    """Overburden extraction gain ``K_e`` (N/m²): lifting a buried foot out costs
    ``K_e * depth²`` (see :func:`granular_extraction_force`). ``0`` (default) = free
    extraction."""
    yield_depth: float = 0.0
    """Penetration beyond which plastic residual accumulates (m)."""
    plastic_rate: float = 0.0
    """Plastic accumulation gain (1/s). ``0`` (default) = no footprint memory."""
    recovery_time: float = 1.0e9
    """Footprint relaxation time constant ``tau_rec`` (s); large = near-permanent."""
    footprint_map_size: tuple[float, float] | None = None
    """When set, residual lives in a world-keyed :class:`FootprintMap` of this ``(x, y)``
    extent (m) instead of a per-foot scalar — feet then couple spatially (ADR-0001 Q3).
    ``None`` (default) = per-foot residual."""
    footprint_map_resolution: float = 0.05
    """Footprint-map cell size (m) when :attr:`footprint_map_size` is set."""
    surface_height: float = 0.0
    """Height of the undisturbed soft surface (m, world z)."""
    foot_link_names: tuple[str, ...] = ()
    """Robot link names whose feet are governed by the analytic model."""
    foot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Foot contact point in the foot *link's* local frame (e.g. a foot site below the
    calf origin). When nonzero, support/traction act at this point (correct lever) via an
    equivalent offset torque, not at the link origin. ``(0, 0, 0)`` = at the link origin."""
    foot_collision_group: int = 0b01
    """``contype``/``conaffinity`` bits for feet (intersect hard terrain, miss soft)."""
    terrain_collision_group: int = 0b10
    """Bits for the soft-terrain geom; disjoint from feet => contact suppressed."""


@dataclass
class TerrainState:
    """Per-environment, per-foot dynamic terrain state (ADR-0001 stage 0 MVP).

    Stage 0 tracks only sinkage ``depth`` and its rate; the residual/shear/recovery
    components arrive in stage 1.
    """

    depth: torch.Tensor
    """Foot penetration below the surface, shape ``(num_envs, num_feet)``."""
    depth_rate: torch.Tensor
    """Rate of change of :attr:`depth`, shape ``(num_envs, num_feet)``."""
    residual: torch.Tensor
    """Plastic residual (footprint memory), shape ``(num_envs, num_feet)``."""
    contact: torch.Tensor
    """Per-foot contact flag (``depth > 0``), shape ``(num_envs, num_feet)`` float."""
    air_time: torch.Tensor
    """Seconds since each foot was last in contact, shape ``(num_envs, num_feet)``."""
    last_air_time: torch.Tensor
    """Air time accumulated just before the latest touchdown (valid where
    :attr:`first_contact`), shape ``(num_envs, num_feet)``."""
    first_contact: torch.Tensor
    """Per-foot flag for the step a foot transitions air→contact, shape ``(E, F)`` float."""
    anchor_xy: torch.Tensor
    """Stick-slip anchor world ``(x, y)`` per foot, shape ``(num_envs, num_feet, 2)``."""

    @classmethod
    def zeros(
        cls, num_envs: int, num_feet: int, device: torch.device | str = "cpu"
    ) -> "TerrainState":
        z = torch.zeros(num_envs, num_feet, device=device)
        return cls(
            depth=z.clone(),
            depth_rate=z.clone(),
            residual=z.clone(),
            contact=z.clone(),
            air_time=z.clone(),
            last_air_time=z.clone(),
            first_contact=z.clone(),
            anchor_xy=torch.zeros(num_envs, num_feet, 2, device=device),
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Zero the state for ``env_ids`` (or all environments when ``None``)."""
        fields = (
            self.depth,
            self.depth_rate,
            self.residual,
            self.contact,
            self.air_time,
            self.last_air_time,
            self.first_contact,
            self.anchor_xy,
        )
        if env_ids is None:
            for t in fields:
                t.zero_()
        else:
            for t in fields:
                t[env_ids] = 0.0


class DeformableTerrain:
    """Genesis-free core of the compliance-only deformable terrain (ADR-0001 stage 0).

    Owns the :class:`TerrainState` and turns foot kinematics into the per-foot support
    force via :func:`compliant_normal_force`. The Genesis runtime wiring (collision-mask
    application and per-step ``apply_links_external_force`` injection, validated in
    ``plans/spikes/``) is layered on top of this core in the integration pass.
    """

    def __init__(
        self,
        cfg: DeformableTerrainCfg,
        num_envs: int,
        num_feet: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.state = TerrainState.zeros(num_envs, num_feet, device=device)
        # Per-env medium parameters, seeded from the cfg scalars. These — not the cfg —
        # are what the force laws read, so domain randomization (e.g.
        # ``mdp.randomize_terrain_params``) and curricula can vary the medium per env
        # by writing the tensors directly.
        self.k = torch.full((num_envs, 1), float(cfg.k), device=device)
        self.mu = torch.full((num_envs, 1), float(cfg.mu), device=device)
        self.drag_coeff = torch.full((num_envs, 1), float(cfg.drag_coeff), device=device)
        self.extraction_coeff = torch.full(
            (num_envs, 1), float(cfg.extraction_coeff), device=device
        )
        self.footprint_map: FootprintMap | None = (
            FootprintMap(
                num_envs, cfg.footprint_map_resolution, cfg.footprint_map_size, device=device
            )
            if cfg.footprint_map_size is not None
            else None
        )

    def compute_normal_force(
        self, foot_height: torch.Tensor, foot_vel_z: torch.Tensor
    ) -> torch.Tensor:
        """Update :attr:`state` from foot kinematics and return the per-foot support force.

        Args:
            foot_height: Contact-point height of each foot (m, world z),
                shape ``(num_envs, num_feet)``.
            foot_vel_z: Vertical velocity of each foot (m/s); negative when descending.
        """
        depth = self.cfg.surface_height - foot_height
        depth_rate = -foot_vel_z
        self.state.depth = depth
        self.state.depth_rate = depth_rate
        # Elastic restoring force references the plastic residual: a foot over an existing
        # footprint (residual > 0) feels less support and rests deeper (paper §6.2, F=k(d-r)).
        elastic_depth = depth - self.state.residual
        return self.cfg.normal_law(elastic_depth, depth_rate, self.k, self.cfg.c)

    def advance_residual(self, dt: float) -> None:
        """Evolve the plastic residual (footprint memory) one step from the stored depth."""
        self.state.residual = plastic_residual_update(
            self.state.residual,
            self.state.depth,
            dt,
            yield_depth=self.cfg.yield_depth,
            plastic_rate=self.cfg.plastic_rate,
            recovery_time=self.cfg.recovery_time,
        )

    def update_gait(self, dt: float) -> None:
        """Update per-foot contact / air-time from the stored depth (terrain-derived gait).

        Contact = ``depth > 0`` (the foot is pressing into the surface). Tracks air time per
        foot and flags the touchdown step, so locomotion rewards can shape a stepping gait
        even though the analytic support produces no *rigid* contact for sensors to read.
        """
        new_contact = (self.state.depth > 0.0).float()
        self.state.first_contact = new_contact * (1.0 - self.state.contact)
        self.state.last_air_time = self.state.air_time.clone()
        self.state.air_time = torch.where(
            new_contact > 0.0, torch.zeros_like(self.state.air_time), self.state.air_time + dt
        )
        self.state.contact = new_contact

    def _stick_slip_tangential(
        self, normal_force: torch.Tensor, foot_pos_xy: torch.Tensor
    ) -> torch.Tensor:
        """Anchored static-friction tangential force per foot (only while in contact)."""
        contact = self.state.contact
        # Fresh, force-free anchor on touchdown or while airborne, so a foot re-anchors at
        # wherever it next lands rather than snapping back to a stale point.
        reset_anchor = (self.state.first_contact > 0.0) | (contact <= 0.0)
        self.state.anchor_xy = torch.where(
            reset_anchor.unsqueeze(-1), foot_pos_xy, self.state.anchor_xy
        )
        shear_max = (
            torch.full_like(normal_force, self.cfg.eta) if self.cfg.eta is not None else None
        )
        f_xy, new_anchor = stick_slip_force(
            self.state.anchor_xy,
            foot_pos_xy,
            normal_force,
            k_lat=self.cfg.lateral_stiffness,
            mu=self.mu,
            shear_max=shear_max,
        )
        self.state.anchor_xy = new_anchor
        return f_xy * contact.unsqueeze(-1)  # only planted feet feel the lateral hold

    def compute_tangential_force(
        self, normal_force: torch.Tensor, slip_velocity: torch.Tensor
    ) -> torch.Tensor:
        """Per-foot Coulomb traction (capped by ``cfg.eta`` shear strength) opposing slip.

        Args:
            normal_force: Per-foot support magnitude, shape ``(num_envs, num_feet)``.
            slip_velocity: Per-foot horizontal velocity, shape ``(num_envs, num_feet, 2)``.

        Returns:
            Tangential force per foot, shape ``(num_envs, num_feet, 2)``.
        """
        shear_max = (
            torch.full_like(normal_force, self.cfg.eta) if self.cfg.eta is not None else None
        )
        return coulomb_tangential_force(normal_force, slip_velocity, self.mu, shear_max=shear_max)

    def apply_foot_forces(
        self,
        solver: Any,
        foot_link_indices: Sequence[int],
        foot_height: torch.Tensor,
        foot_vel_z: torch.Tensor,
        foot_vel_xy: torch.Tensor | None = None,
        foot_pos_xy: torch.Tensor | None = None,
        offset_w: torch.Tensor | None = None,
        dt: float = 0.0,
    ) -> torch.Tensor:
        """Inject the analytic support (and traction) at the foot links via the rigid solver.

        Validated against the real solver in ``plans/spikes/analytic_sinkage_spike.py``:
        with no rigid floor under the feet, this world-frame force is the foot's only
        support. The normal component settles the foot at ``d_eq = W / k``; when
        ``foot_vel_xy`` is supplied and ``cfg.mu > 0`` the horizontal component adds
        Coulomb traction (capped by ``cfg.eta``) so the foot gets grip — the stage-1
        ingredient that lets a robot walk on soft terrain.

        Args:
            solver: Genesis ``RigidSolver`` (``link.solver``) exposing
                ``apply_links_external_force``.
            foot_link_indices: Genesis link indices of the feet, ordered to match the
                ``num_feet`` axis of ``foot_height`` / ``foot_vel_z``.
            foot_height: Per-foot contact-point height, shape ``(num_envs, num_feet)``.
            foot_vel_z: Per-foot vertical velocity, shape ``(num_envs, num_feet)``.
            foot_vel_xy: Optional per-foot horizontal velocity, shape
                ``(num_envs, num_feet, 2)``. Omitted (or ``cfg.mu == 0``) = normal-only.
            foot_pos_xy: Optional per-foot world ``(x, y)`` position, shape
                ``(num_envs, num_feet, 2)``. Required to key the spatial footprint map.
            dt: Physics timestep (s). When ``> 0`` and ``cfg.plastic_rate > 0`` the plastic
                residual (footprint memory) is advanced one step after the force is read.

        Returns:
            The injected world-frame force tensor, shape ``(num_envs, num_feet, 3)``.
        """
        spatial = self.footprint_map is not None and foot_pos_xy is not None
        if spatial:
            # Residual at the feet comes from the world-keyed map: feet couple spatially.
            self.state.residual = self.footprint_map.read(foot_pos_xy)  # type: ignore[union-attr,arg-type]
        fz = self.compute_normal_force(foot_height, foot_vel_z)
        if dt > 0.0:
            self.update_gait(dt)  # contact / air-time before the tangential model reads them
        force = torch.zeros(*fz.shape, 3, device=fz.device)
        force[..., 2] = fz
        if self.cfg.lateral_stiffness > 0.0 and foot_pos_xy is not None:
            force[..., :2] = self._stick_slip_tangential(fz, foot_pos_xy)
        elif foot_vel_xy is not None and self.cfg.mu > 0.0:
            force[..., :2] = self.compute_tangential_force(fz, foot_vel_xy)
        if foot_vel_xy is not None and bool((self.drag_coeff > 0.0).any()):
            # Plowing drag is the medium's own failure resistance — additive to the
            # contact traction and NOT capped by mu*F_z (it exists at zero load).
            force[..., :2] += granular_drag_force(self.state.depth, foot_vel_xy, self.drag_coeff)
        if bool((self.extraction_coeff > 0.0).any()):
            force[..., 2] += granular_extraction_force(
                self.state.depth, foot_vel_z, self.extraction_coeff
            )
        solver.apply_links_external_force(force, foot_link_indices)
        if offset_w is not None:
            # Apply the force at the foot point (offset from the link origin) by adding the
            # equivalent torque ``offset × force`` — gives the correct lever for stance.
            torque = torch.cross(offset_w, force, dim=-1)
            solver.apply_links_external_torque(torque, foot_link_indices)
        if dt > 0.0 and self.cfg.plastic_rate > 0.0:
            if spatial:
                accumulation = (
                    self.cfg.plastic_rate
                    * (self.state.depth - self.cfg.yield_depth).clamp_min(0.0)
                    * dt
                )
                self.footprint_map.accumulate(foot_pos_xy, accumulation)  # type: ignore[union-attr,arg-type]
                self.footprint_map.recover(dt, self.cfg.recovery_time)  # type: ignore[union-attr]
            else:
                self.advance_residual(dt)
        return force


def _as_batched(value: Any, num_envs: int, device: torch.device | str) -> torch.Tensor:
    """Coerce a Genesis links getter result to ``(num_envs, num_links, 3)`` on ``device``."""
    tensor = torch.as_tensor(value, device=device)
    if tensor.dim() == 2:  # single-env builds may return the bare (num_links, 3)
        tensor = tensor.unsqueeze(0).expand(num_envs, -1, -1)
    return tensor


class DeformableTerrainDriver:
    """Binds a :class:`DeformableTerrain` to a live articulation and drives it per step.

    The Genesis-touching layer around the pure :class:`DeformableTerrain`: resolves the
    foot link indices once at construction, then each physics tick reads the feet's world
    height/velocity and injects the analytic support force. The env owns one of these and
    calls :meth:`apply` inside its decimation loop (before each ``scene.step``) and
    :meth:`reset` on environment reset.
    """

    def __init__(
        self,
        cfg: DeformableTerrainCfg,
        articulation: Any,
        num_envs: int,
        device: torch.device | str = "cpu",
        dt: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self._num_envs = num_envs
        self._device = device
        self._dt = dt
        self._gs_handle = articulation.gs_handle
        link_names = list(articulation.link_names)
        self._foot_local_idx = [link_names.index(name) for name in cfg.foot_link_names]
        links = self._gs_handle.links
        self._foot_global_idx = tuple(links[i].idx for i in self._foot_local_idx)
        # All links share the one rigid solver; grab it from the first foot.
        self._solver = links[self._foot_local_idx[0]].solver if self._foot_local_idx else None
        self.terrain = DeformableTerrain(cfg, num_envs, len(self._foot_local_idx), device=device)

    def apply(self) -> None:
        """Read foot kinematics and inject the analytic support (per physics tick)."""
        if self._solver is None:
            return
        pos = _as_batched(self._gs_handle.get_links_pos(), self._num_envs, self._device)
        vel = _as_batched(self._gs_handle.get_links_vel(), self._num_envs, self._device)
        link_pos = pos[:, self._foot_local_idx, :]
        link_vel = vel[:, self._foot_local_idx, :]
        offset_w: torch.Tensor | None = None
        if any(self.cfg.foot_offset):
            # Resolve the contact point at the foot *site* (offset in the link's frame),
            # carried by the leg's orientation, and the velocity there — so support/traction
            # act at the true foot point. ``offset_w`` carries the equivalent torque.
            from genelab.utils.math import quat_apply

            quat = _as_batched(self._gs_handle.get_links_quat(), self._num_envs, self._device)
            ang = _as_batched(self._gs_handle.get_links_ang(), self._num_envs, self._device)
            link_quat = quat[:, self._foot_local_idx, :]
            link_ang = ang[:, self._foot_local_idx, :]
            local_offset = torch.tensor(self.cfg.foot_offset, device=self._device)
            offset_w = quat_apply(link_quat, local_offset.expand_as(link_pos))
            foot_pos = link_pos + offset_w
            foot_vel = link_vel + torch.cross(link_ang, offset_w, dim=-1)
        else:
            foot_pos = link_pos
            foot_vel = link_vel
        self.terrain.apply_foot_forces(
            self._solver,
            self._foot_global_idx,
            foot_pos[..., 2],
            foot_vel[..., 2],
            foot_vel[..., :2],
            foot_pos_xy=foot_pos[..., :2],
            offset_w=offset_w,
            dt=self._dt,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.terrain.state.reset(env_ids)
