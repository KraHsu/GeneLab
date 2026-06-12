"""Unit tests for ``genelab.terrains.deformable`` analytic force laws (ADR-0001 stage 0).

Pure tensor seams — no Genesis. The force law is the deep core of the compliance-only
degenerate branch validated end to end in ``plans/spikes/analytic_sinkage_spike.py``.
"""

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.terrains.deformable import (  # noqa: E402
    DeformableTerrain,
    DeformableTerrainCfg,
    DeformableTerrainDriver,
    TerrainState,
    compliant_normal_force,
    coulomb_tangential_force,
    granular_drag_force,
    granular_extraction_force,
    plastic_residual_update,
    stick_slip_force,
)


def test_elastic_support_at_rest_is_stiffness_times_depth() -> None:
    f = compliant_normal_force(
        depth=torch.tensor(0.05), depth_rate=torch.tensor(0.0), k=1000.0, c=140.0
    )
    assert float(f) == pytest.approx(50.0)


def test_foot_above_surface_feels_no_force() -> None:
    f = compliant_normal_force(
        depth=torch.tensor(-0.02), depth_rate=torch.tensor(0.0), k=1000.0, c=140.0
    )
    assert float(f) == pytest.approx(0.0)


def test_sinking_adds_damping_force() -> None:
    f = compliant_normal_force(
        depth=torch.tensor(0.05), depth_rate=torch.tensor(0.1), k=1000.0, c=140.0
    )
    assert float(f) == pytest.approx(64.0)  # 1000*0.05 + 140*0.1


def test_fast_rebound_never_pulls_the_foot_down() -> None:
    # In light contact but rebounding fast: k*d + c*ḋ goes negative -> clamped to 0.
    f = compliant_normal_force(
        depth=torch.tensor(0.001), depth_rate=torch.tensor(-5.0), k=1000.0, c=140.0
    )
    assert float(f) == pytest.approx(0.0)


def test_force_is_evaluated_per_foot() -> None:
    f = compliant_normal_force(
        depth=torch.tensor([0.05, -0.02, 0.01]),
        depth_rate=torch.tensor([0.0, 0.0, 0.1]),
        k=1000.0,
        c=140.0,
    )
    assert torch.allclose(f, torch.tensor([50.0, 0.0, 24.0]))


# ------------------------------------------------------ tangential friction law


def test_tangential_force_opposes_slip_at_the_coulomb_limit() -> None:
    f = coulomb_tangential_force(
        normal_force=torch.tensor(100.0),
        slip_velocity=torch.tensor([0.5, 0.0]),  # sliding in +x, well above the reg. eps
        mu=0.8,
    )
    assert float(f[0]) < 0.0  # opposes the +x slip
    assert float(f[1]) == pytest.approx(0.0)
    assert float(f.norm()) == pytest.approx(80.0, rel=0.02)  # |F| -> mu * F_z


def test_tangential_force_vanishes_as_the_foot_stops() -> None:
    f = coulomb_tangential_force(
        normal_force=torch.tensor(100.0), slip_velocity=torch.tensor([0.0, 0.0]), mu=0.8
    )
    assert float(f.norm()) == pytest.approx(0.0)


def test_tangential_force_is_evaluated_per_foot() -> None:
    f = coulomb_tangential_force(
        normal_force=torch.tensor([100.0, 50.0]),
        slip_velocity=torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        mu=0.5,
    )
    assert f.shape == (2, 2)
    assert float(f[0, 0]) == pytest.approx(-50.0, rel=0.02)  # foot 0: -mu*100 in x
    assert float(f[1, 1]) == pytest.approx(-25.0, rel=0.02)  # foot 1: -mu*50 in y


def test_stick_slip_force_anchors_a_planted_foot_at_zero_velocity() -> None:
    # A foot displaced 1 cm from its anchor, well under the Coulomb cap, feels a restoring
    # (spring) force pulling it back — static friction, present at ZERO slip velocity.
    force, new_anchor = stick_slip_force(
        anchor_xy=torch.tensor([0.0, 0.0]),
        foot_pos_xy=torch.tensor([0.01, 0.0]),
        normal_force=torch.tensor(100.0),
        k_lat=2000.0,
        mu=0.8,
    )
    assert float(force[0]) == pytest.approx(-20.0)  # -k_lat * disp, opposes displacement
    assert float(force[1]) == pytest.approx(0.0)
    assert torch.allclose(new_anchor, torch.tensor([0.0, 0.0]), atol=1e-6)  # stuck: anchor holds


def test_stick_slip_force_saturates_at_the_cap_and_drags_the_anchor() -> None:
    # 10 cm displacement: spring force -200 N exceeds the Coulomb cap mu*F_z = 80 -> slip.
    force, new_anchor = stick_slip_force(
        anchor_xy=torch.tensor([0.0, 0.0]),
        foot_pos_xy=torch.tensor([0.1, 0.0]),
        normal_force=torch.tensor(100.0),
        k_lat=2000.0,
        mu=0.8,
    )
    assert float(torch.linalg.norm(force)) == pytest.approx(80.0, rel=1e-3)  # capped at mu*F_z
    assert float(force[0]) < 0.0  # still opposes displacement
    # anchor dragged so the (capped) force is consistent with the new anchor.
    assert torch.allclose(force, -2000.0 * (torch.tensor([0.1, 0.0]) - new_anchor), atol=1e-2)


def test_stick_slip_force_is_zero_at_a_fresh_anchor() -> None:
    force, _ = stick_slip_force(
        anchor_xy=torch.tensor([0.05, 0.02]),
        foot_pos_xy=torch.tensor([0.05, 0.02]),
        normal_force=torch.tensor(100.0),
        k_lat=2000.0,
        mu=0.8,
    )
    assert float(torch.linalg.norm(force)) == pytest.approx(0.0)


def test_shear_cap_limits_traction_below_the_coulomb_limit() -> None:
    # Granular shear strength caps traction: a shear_max (20 N) below mu*F_z (80 N) wins.
    f = coulomb_tangential_force(
        normal_force=torch.tensor(100.0),
        slip_velocity=torch.tensor([0.5, 0.0]),
        mu=0.8,
        shear_max=torch.tensor(20.0),
    )
    assert float(f.norm()) == pytest.approx(20.0, rel=0.02)


def test_granular_drag_opposes_motion_at_depth_squared_magnitude() -> None:
    # Plowing through granular media: F = K_d * depth^2, independent of normal load.
    # At speeds well above the regularization knee the magnitude is the full K_d*d^2.
    f = granular_drag_force(
        depth=torch.tensor(0.1),
        velocity_xy=torch.tensor([5.0, 0.0]),
        drag_coeff=1500.0,
    )
    assert float(f[0]) < 0.0  # opposes the +x motion
    assert float(f[1]) == pytest.approx(0.0)
    assert float(f.norm()) == pytest.approx(1500.0 * 0.1**2, rel=0.02)


def test_granular_drag_quadruples_when_depth_doubles() -> None:
    shallow = granular_drag_force(
        depth=torch.tensor(0.05), velocity_xy=torch.tensor([5.0, 0.0]), drag_coeff=1500.0
    )
    deep = granular_drag_force(
        depth=torch.tensor(0.1), velocity_xy=torch.tensor([5.0, 0.0]), drag_coeff=1500.0
    )
    assert float(deep.norm()) == pytest.approx(4.0 * float(shallow.norm()), rel=1e-5)


def test_granular_drag_vanishes_above_the_surface() -> None:
    f = granular_drag_force(
        depth=torch.tensor(-0.02),  # foot above the sand: nothing to plow
        velocity_xy=torch.tensor([0.5, 0.0]),
        drag_coeff=1500.0,
    )
    assert float(f.norm()) == pytest.approx(0.0)


def test_granular_drag_vanishes_as_the_foot_stops() -> None:
    # Regularized at v -> 0: no chattering force on a foot at rest in the sand.
    f = granular_drag_force(
        depth=torch.tensor(0.1),
        velocity_xy=torch.tensor([1.0e-5, 0.0]),
        drag_coeff=1500.0,
    )
    assert float(f.norm()) < 0.01


def test_granular_drag_is_evaluated_per_foot() -> None:
    f = granular_drag_force(
        depth=torch.tensor([[0.1, 0.0]]),  # foot 0 buried, foot 1 at the surface
        velocity_xy=torch.tensor([[[5.0, 0.0], [5.0, 0.0]]]),
        drag_coeff=1500.0,
    )
    assert f.shape == (1, 2, 2)
    assert float(f[0, 0].norm()) == pytest.approx(15.0, rel=0.02)
    assert float(f[0, 1].norm()) == pytest.approx(0.0)


def test_granular_extraction_resists_only_upward_motion() -> None:
    # Pulling a buried foot up mobilizes the overburden: F_z = -K_e * d^2 against the
    # lift. Moving down is the compliance/damping model's regime — no extraction force.
    up = granular_extraction_force(
        depth=torch.tensor(0.1), foot_vel_z=torch.tensor(5.0), extraction_coeff=500.0
    )
    down = granular_extraction_force(
        depth=torch.tensor(0.1), foot_vel_z=torch.tensor(-5.0), extraction_coeff=500.0
    )
    assert float(up) == pytest.approx(-500.0 * 0.1**2, rel=0.02)
    assert float(down) == pytest.approx(0.0)


def test_granular_extraction_vanishes_above_the_surface() -> None:
    f = granular_extraction_force(
        depth=torch.tensor(-0.02), foot_vel_z=torch.tensor(1.0), extraction_coeff=500.0
    )
    assert float(f) == pytest.approx(0.0)


def test_granular_extraction_vanishes_as_the_foot_stops() -> None:
    f = granular_extraction_force(
        depth=torch.tensor(0.1), foot_vel_z=torch.tensor(1.0e-5), extraction_coeff=500.0
    )
    assert abs(float(f)) < 0.01


def test_granular_extraction_is_evaluated_per_foot() -> None:
    f = granular_extraction_force(
        depth=torch.tensor([[0.1, 0.1]]),
        foot_vel_z=torch.tensor([[5.0, -5.0]]),  # foot 0 lifting, foot 1 descending
        extraction_coeff=500.0,
    )
    assert f.shape == (1, 2)
    assert float(f[0, 0]) == pytest.approx(-5.0, rel=0.02)
    assert float(f[0, 1]) == pytest.approx(0.0)


# --------------------------------------------------- plastic residual (memory)


def test_plastic_residual_accumulates_past_the_yield_depth() -> None:
    r = plastic_residual_update(
        residual=torch.tensor(0.0),
        depth=torch.tensor(0.05),  # 0.03 past the yield depth
        dt=0.01,
        yield_depth=0.02,
        plastic_rate=1.0,
        recovery_time=1.0e9,  # effectively no recovery this step
    )
    assert float(r) == pytest.approx(1.0 * (0.05 - 0.02) * 0.01)  # plastic_rate*excess*dt


def test_plastic_residual_recovers_toward_zero_below_yield() -> None:
    r = plastic_residual_update(
        residual=torch.tensor(0.01),
        depth=torch.tensor(0.0),  # below yield -> no accumulation, only recovery
        dt=0.1,
        yield_depth=0.02,
        plastic_rate=1.0,
        recovery_time=1.0,
    )
    assert float(r) == pytest.approx(0.009)  # 0.01 - (0.1/1.0)*0.01


# ----------------------------------------------------------------- TerrainState


def test_terrain_state_starts_zeroed_with_per_env_per_foot_shape() -> None:
    state = TerrainState.zeros(num_envs=4, num_feet=2)
    assert state.depth.shape == (4, 2)
    assert state.depth_rate.shape == (4, 2)
    assert float(state.depth.abs().sum()) == 0.0


def test_terrain_state_tracks_plastic_residual() -> None:
    state = TerrainState.zeros(num_envs=2, num_feet=3)
    assert state.residual.shape == (2, 3)
    assert float(state.residual.abs().sum()) == 0.0
    state.residual += 0.02
    state.reset()
    assert float(state.residual.abs().sum()) == 0.0


def test_terrain_state_reset_clears_only_selected_envs() -> None:
    state = TerrainState.zeros(num_envs=3, num_feet=2)
    state.depth += 0.05
    state.reset(env_ids=torch.tensor([0, 2]))
    assert float(state.depth[0].sum()) == 0.0
    assert float(state.depth[2].sum()) == 0.0
    assert torch.allclose(state.depth[1], torch.full((2,), 0.05))


# ------------------------------------------------------------- DeformableTerrain


def test_foot_below_surface_gets_upward_support_and_records_sinkage() -> None:
    cfg = DeformableTerrainCfg(k=1000.0, c=0.0, surface_height=0.0)
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    # Foot 3 cm below the surface, momentarily still.
    f = terrain.compute_normal_force(
        foot_height=torch.tensor([[-0.03]]), foot_vel_z=torch.tensor([[0.0]])
    )
    assert float(f) == pytest.approx(30.0)  # k * depth = 1000 * 0.03
    assert float(terrain.state.depth) == pytest.approx(0.03)


def test_model_tangential_force_uses_cfg_friction() -> None:
    cfg = DeformableTerrainCfg(mu=0.5)
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    f = terrain.compute_tangential_force(
        normal_force=torch.tensor([[100.0]]),
        slip_velocity=torch.tensor([[[1.0, 0.0]]]),  # (1, 1, 2): sliding in +x
    )
    assert f.shape == (1, 1, 2)
    assert float(f[0, 0, 0]) == pytest.approx(-50.0, rel=0.02)  # -mu * F_z


def test_normal_force_references_the_plastic_residual() -> None:
    cfg = DeformableTerrainCfg(k=1000.0, c=0.0, surface_height=0.0)
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    terrain.state.residual = torch.tensor([[0.01]])  # a 1 cm footprint already set
    f = terrain.compute_normal_force(
        foot_height=torch.tensor([[-0.05]]),  # depth 0.05; elastic ref = 0.05 - 0.01
        foot_vel_z=torch.tensor([[0.0]]),
    )
    assert float(f) == pytest.approx(40.0)  # k * (depth - residual)


def test_default_normal_law_is_the_linear_compliant_force() -> None:
    cfg = DeformableTerrainCfg()
    assert cfg.normal_law is compliant_normal_force


def test_custom_normal_law_replaces_the_support_model() -> None:
    # A pneumatic-style law: stiffens as the foot approaches the 0.2 m bottom-out depth.
    def pneumatic(depth, depth_rate, k, c):  # type: ignore[no-untyped-def]
        del depth_rate, c
        compression = (depth.clamp_min(0.0) / 0.2).clamp_max(0.99)
        return k * 0.2 * compression / (1.0 - compression)

    cfg = DeformableTerrainCfg(k=1000.0, c=0.0, normal_law=pneumatic)
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=2)
    f = terrain.compute_normal_force(
        foot_height=torch.tensor([[-0.1, -0.18]]),  # half vs near bottom-out
        foot_vel_z=torch.zeros(1, 2),
    )
    # Half compression: 1000*0.2*0.5/0.5 = 200 N; near bottom-out blows up past linear.
    assert float(f[0, 0]) == pytest.approx(200.0)
    assert float(f[0, 1]) > 1000.0 * 0.18  # far stiffer than the linear law would give


def test_custom_normal_law_still_references_the_plastic_residual() -> None:
    def linear_no_clamp(depth, depth_rate, k, c):  # type: ignore[no-untyped-def]
        del depth_rate, c
        return k * depth

    cfg = DeformableTerrainCfg(k=1000.0, c=0.0, normal_law=linear_no_clamp)
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    terrain.state.residual = torch.tensor([[0.02]])  # an existing 2 cm footprint
    f = terrain.compute_normal_force(
        foot_height=torch.tensor([[-0.05]]), foot_vel_z=torch.zeros(1, 1)
    )
    assert float(f[0, 0]) == pytest.approx(1000.0 * (0.05 - 0.02))


def test_model_advance_residual_accumulates_footprint_from_depth() -> None:
    cfg = DeformableTerrainCfg(yield_depth=0.01, plastic_rate=2.0, recovery_time=1.0e9)
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    terrain.state.depth = torch.tensor([[0.05]])  # 0.04 past the yield depth
    terrain.advance_residual(dt=0.1)
    assert float(terrain.state.residual) == pytest.approx(2.0 * 0.04 * 0.1)  # rate*excess*dt


def test_foot_resting_on_surface_gets_no_support() -> None:
    cfg = DeformableTerrainCfg(k=1000.0, c=0.0, surface_height=0.0)
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    f = terrain.compute_normal_force(
        foot_height=torch.tensor([[0.0]]), foot_vel_z=torch.tensor([[0.0]])
    )
    assert float(f) == pytest.approx(0.0)


def test_soft_terrain_collision_group_is_disjoint_from_feet() -> None:
    # Spike-B invariant: disjoint masks => the foot<->soft-terrain contact is bypassed.
    cfg = DeformableTerrainCfg()
    assert cfg.foot_collision_group & cfg.terrain_collision_group == 0


# ------------------------------------------------------- injection seam (fake gs)


class _FakeSolver:
    """Records ``apply_links_external_force`` calls without a Genesis runtime."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []
        self.torque_calls: list[tuple[Any, Any]] = []

    def apply_links_external_force(
        self,
        force: Any,
        links_idx: Any = None,
        envs_idx: Any = None,  # noqa: ARG002 - mirrors the Genesis signature
        **kw: Any,
    ) -> None:
        self.calls.append((force, links_idx))

    def apply_links_external_torque(
        self,
        torque: Any,
        links_idx: Any = None,
        envs_idx: Any = None,  # noqa: ARG002 - mirrors the Genesis signature
        **kw: Any,
    ) -> None:
        self.torque_calls.append((torque, links_idx))


def test_apply_foot_forces_injects_vertical_support_at_the_foot_links() -> None:
    cfg = DeformableTerrainCfg(k=1000.0, c=0.0, surface_height=0.0)
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    solver = _FakeSolver()

    terrain.apply_foot_forces(
        solver,
        foot_link_indices=(3,),
        foot_height=torch.tensor([[-0.03]]),  # 3 cm below surface
        foot_vel_z=torch.tensor([[0.0]]),
    )

    assert len(solver.calls) == 1
    force, links_idx = solver.calls[0]
    assert links_idx == (3,)
    assert force.shape == (1, 1, 3)
    assert float(force[0, 0, 2]) == pytest.approx(30.0)  # k*depth up
    assert float(force[0, 0, 0]) == 0.0  # no tangential without slip input
    assert float(force[0, 0, 1]) == 0.0


def test_apply_foot_forces_injects_traction_opposing_slip() -> None:
    cfg = DeformableTerrainCfg(k=1000.0, c=0.0, mu=0.5, surface_height=0.0)
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    solver = _FakeSolver()

    terrain.apply_foot_forces(
        solver,
        foot_link_indices=(3,),
        foot_height=torch.tensor([[-0.04]]),  # depth 0.04 -> F_z = 40
        foot_vel_z=torch.tensor([[0.0]]),
        foot_vel_xy=torch.tensor([[[0.3, 0.0]]]),  # sliding +x
    )

    force, _ = solver.calls[0]
    assert float(force[0, 0, 2]) == pytest.approx(40.0)  # normal support unchanged
    assert float(force[0, 0, 0]) == pytest.approx(-20.0, rel=0.05)  # -mu*F_z, opposes slip
    assert float(force[0, 0, 1]) == pytest.approx(0.0)


# ------------------------------------------------ driver binding (fake articulation)


class _FakeLink:
    def __init__(self, idx: int, solver: Any) -> None:
        self.idx = idx
        self.solver = solver


class _FakeGsHandle:
    def __init__(self, links: list[_FakeLink], pos: Any, vel: Any) -> None:
        self.links = links
        self._pos = pos
        self._vel = vel

    def get_links_pos(self) -> Any:
        return self._pos

    def get_links_vel(self) -> Any:
        return self._vel


class _FakeArticulation:
    def __init__(self, link_names: list[str], gs_handle: _FakeGsHandle) -> None:
        self.link_names = link_names
        self.gs_handle = gs_handle


def _quadruped_stub(
    foot_z: list[float],
    foot_vz: list[float],
    solver: Any,
    foot_vxy: list[tuple[float, float]] | None = None,
) -> _FakeArticulation:
    # 4 links: base + two feet + trunk; feet are local indices 1 and 2, global idx 11 and 12.
    link_names = ["base", "FL_foot", "FR_foot", "trunk"]
    links = [_FakeLink(idx, solver) for idx in (10, 11, 12, 13)]
    pos = torch.zeros(1, 4, 3)
    pos[0, 1, 2], pos[0, 2, 2] = foot_z
    vel = torch.zeros(1, 4, 3)
    vel[0, 1, 2], vel[0, 2, 2] = foot_vz
    if foot_vxy is not None:
        vel[0, 1, 0], vel[0, 1, 1] = foot_vxy[0]
        vel[0, 2, 0], vel[0, 2, 1] = foot_vxy[1]
    return _FakeArticulation(link_names, _FakeGsHandle(links, pos, vel))


def test_driver_injects_per_foot_support_at_resolved_global_link_indices() -> None:
    solver = _FakeSolver()
    art = _quadruped_stub(foot_z=[-0.03, -0.01], foot_vz=[0.0, 0.0], solver=solver)
    cfg = DeformableTerrainCfg(
        k=1000.0, c=0.0, surface_height=0.0, foot_link_names=("FL_foot", "FR_foot")
    )
    driver = DeformableTerrainDriver(cfg, art, num_envs=1, device="cpu")

    driver.apply()

    force, links_idx = solver.calls[0]
    assert links_idx == (11, 12)  # global indices of FL_foot, FR_foot
    assert force.shape == (1, 2, 3)
    assert torch.allclose(force[0, :, 2], torch.tensor([30.0, 10.0]))  # k*depth per foot
    # Sinkage recorded per foot in the owned state.
    assert torch.allclose(driver.terrain.state.depth[0], torch.tensor([0.03, 0.01]))


def test_apply_foot_forces_advances_residual_over_repeated_presses() -> None:
    cfg = DeformableTerrainCfg(
        k=1000.0, c=0.0, yield_depth=0.01, plastic_rate=5.0, recovery_time=1.0e9
    )
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    solver = _FakeSolver()
    for _ in range(3):  # press 0.05 deep three times: excess 0.04, +5*0.04*0.1 = 0.02 each
        terrain.apply_foot_forces(
            solver,
            foot_link_indices=(3,),
            foot_height=torch.tensor([[-0.05]]),
            foot_vel_z=torch.tensor([[0.0]]),
            dt=0.1,
        )
    assert float(terrain.state.residual) == pytest.approx(0.06, rel=0.02)


def test_apply_foot_forces_writes_plastic_into_the_spatial_footprint_map() -> None:
    cfg = DeformableTerrainCfg(
        k=1000.0,
        c=0.0,
        yield_depth=0.01,
        plastic_rate=5.0,
        recovery_time=1.0e9,
        footprint_map_size=(2.0, 2.0),
        footprint_map_resolution=0.1,
    )
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    solver = _FakeSolver()
    foot_xy = torch.tensor([[[0.3, 0.4]]])  # (env, foot, 2)

    terrain.apply_foot_forces(
        solver,
        foot_link_indices=(3,),
        foot_height=torch.tensor([[-0.05]]),
        foot_vel_z=torch.tensor([[0.0]]),
        foot_pos_xy=foot_xy,
        dt=0.1,
    )

    assert terrain.footprint_map is not None
    assert float(terrain.footprint_map.read(foot_xy)) > 0.0  # footprint stamped here
    assert float(terrain.footprint_map.read(torch.tensor([[[-0.5, -0.5]]]))) == 0.0


def test_force_reads_spatial_residual_from_the_footprint_map() -> None:
    cfg = DeformableTerrainCfg(
        k=1000.0,
        c=0.0,
        surface_height=0.0,
        footprint_map_size=(2.0, 2.0),
        footprint_map_resolution=0.1,
    )
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    assert terrain.footprint_map is not None
    foot_xy = torch.tensor([[[0.3, 0.4]]])
    terrain.footprint_map.accumulate(foot_xy, torch.tensor([[0.01]]))  # a 1 cm footprint here
    solver = _FakeSolver()

    terrain.apply_foot_forces(
        solver,
        foot_link_indices=(3,),
        foot_height=torch.tensor([[-0.05]]),  # depth 0.05
        foot_vel_z=torch.tensor([[0.0]]),
        foot_pos_xy=foot_xy,
    )

    force, _ = solver.calls[0]
    assert float(force[0, 0, 2]) == pytest.approx(40.0)  # k*(0.05 - 0.01 map residual)


def test_apply_foot_forces_applies_offset_torque_at_the_foot_point() -> None:
    cfg = DeformableTerrainCfg(k=1000.0, c=0.0, surface_height=0.0)
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    solver = _FakeSolver()
    offset_w = torch.tensor([[[0.1, 0.0, -0.2]]])  # foot point relative to link origin (world)

    terrain.apply_foot_forces(
        solver,
        foot_link_indices=(3,),
        foot_height=torch.tensor([[-0.05]]),  # F_z = 50 up
        foot_vel_z=torch.tensor([[0.0]]),
        offset_w=offset_w,
    )

    torque, idx = solver.torque_calls[0]
    # Equivalent torque for applying F at the offset point: offset × F = (0.1,0,-0.2)×(0,0,50).
    assert torch.allclose(torque[0, 0], torch.tensor([0.0, -5.0, 0.0]), atol=1e-4)
    assert idx == (3,)


def test_apply_foot_forces_stick_slip_gives_a_static_lateral_hold() -> None:
    cfg = DeformableTerrainCfg(
        k=1000.0, c=0.0, mu=0.8, surface_height=0.0, lateral_stiffness=2000.0
    )
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    solver = _FakeSolver()
    # Step 1: foot lands at x=0 (depth 0.04) -> establishes the anchor, ~0 lateral force.
    terrain.apply_foot_forces(
        solver,
        foot_link_indices=(3,),
        foot_height=torch.tensor([[-0.04]]),
        foot_vel_z=torch.tensor([[0.0]]),
        foot_pos_xy=torch.tensor([[[0.0, 0.0]]]),
        dt=0.02,
    )
    # Step 2: same foot now at x=0.01, still planted, NO velocity passed -> restoring force.
    terrain.apply_foot_forces(
        solver,
        foot_link_indices=(3,),
        foot_height=torch.tensor([[-0.04]]),
        foot_vel_z=torch.tensor([[0.0]]),
        foot_pos_xy=torch.tensor([[[0.01, 0.0]]]),
        dt=0.02,
    )
    force, _ = solver.calls[-1]
    assert float(force[0, 0, 2]) == pytest.approx(40.0)  # normal support unchanged
    assert float(force[0, 0, 0]) == pytest.approx(
        -20.0, rel=0.05
    )  # -k_lat*disp, zero-velocity hold


def test_apply_foot_forces_adds_plowing_drag_on_top_of_stick_slip() -> None:
    cfg = DeformableTerrainCfg(
        k=1000.0,
        c=0.0,
        mu=0.8,
        surface_height=0.0,
        lateral_stiffness=2000.0,
        drag_coeff=1500.0,
    )
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    solver = _FakeSolver()
    # Fresh anchor at the foot -> zero stick-slip force; the buried foot sweeping +x at
    # high speed feels the full plowing drag K_d * d^2 (independent of mu and F_z).
    terrain.apply_foot_forces(
        solver,
        foot_link_indices=(3,),
        foot_height=torch.tensor([[-0.1]]),
        foot_vel_z=torch.tensor([[0.0]]),
        foot_vel_xy=torch.tensor([[[5.0, 0.0]]]),
        foot_pos_xy=torch.tensor([[[0.0, 0.0]]]),
        dt=0.02,
    )
    force, _ = solver.calls[-1]
    assert float(force[0, 0, 0]) == pytest.approx(-15.0, rel=0.05)  # -K_d*d^2, opposes sweep
    assert float(force[0, 0, 1]) == pytest.approx(0.0)


def test_apply_foot_forces_resists_extracting_a_buried_foot() -> None:
    cfg = DeformableTerrainCfg(k=1000.0, c=0.0, surface_height=0.0, extraction_coeff=500.0)
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    solver = _FakeSolver()
    # Foot buried 0.1 m, lifting fast: support k*d=100 minus extraction K_e*d^2=5.
    terrain.apply_foot_forces(
        solver,
        foot_link_indices=(3,),
        foot_height=torch.tensor([[-0.1]]),
        foot_vel_z=torch.tensor([[5.0]]),
    )
    force, _ = solver.calls[-1]
    assert float(force[0, 0, 2]) == pytest.approx(100.0 - 5.0, rel=0.02)


def test_medium_params_are_per_env_tensors_seeded_from_cfg() -> None:
    cfg = DeformableTerrainCfg(k=1000.0, mu=0.5, drag_coeff=300.0, extraction_coeff=100.0)
    terrain = DeformableTerrain(cfg, num_envs=3, num_feet=2)
    for field, expected in (
        ("k", 1000.0),
        ("mu", 0.5),
        ("drag_coeff", 300.0),
        ("extraction_coeff", 100.0),
    ):
        t = getattr(terrain, field)
        assert t.shape == (3, 1)
        assert torch.allclose(t, torch.full((3, 1), expected))


def test_per_env_stiffness_gives_per_env_support() -> None:
    cfg = DeformableTerrainCfg(k=1000.0, c=0.0, surface_height=0.0)
    terrain = DeformableTerrain(cfg, num_envs=2, num_feet=1)
    terrain.k[1] = 2000.0  # env 1 is twice as firm
    f = terrain.compute_normal_force(
        foot_height=torch.tensor([[-0.05], [-0.05]]), foot_vel_z=torch.zeros(2, 1)
    )
    assert float(f[0, 0]) == pytest.approx(50.0)
    assert float(f[1, 0]) == pytest.approx(100.0)


def test_per_env_drag_gives_per_env_plowing_resistance() -> None:
    cfg = DeformableTerrainCfg(k=1000.0, c=0.0, surface_height=0.0, drag_coeff=1500.0)
    terrain = DeformableTerrain(cfg, num_envs=2, num_feet=1)
    terrain.drag_coeff[1] = 0.0  # env 1's medium has no plowing resistance
    solver = _FakeSolver()
    terrain.apply_foot_forces(
        solver,
        foot_link_indices=(3,),
        foot_height=torch.tensor([[-0.1], [-0.1]]),
        foot_vel_z=torch.zeros(2, 1),
        foot_vel_xy=torch.tensor([[[5.0, 0.0]], [[5.0, 0.0]]]),
    )
    force, _ = solver.calls[-1]
    assert float(force[0, 0, 0]) == pytest.approx(-15.0, rel=0.05)
    assert float(force[1, 0, 0]) == pytest.approx(0.0)


def test_randomize_terrain_params_resamples_only_selected_envs() -> None:
    from genelab.mdp.events import randomize_terrain_params

    cfg = DeformableTerrainCfg(k=1000.0, foot_link_names=("FL_foot", "FR_foot"))
    solver = _FakeSolver()
    art = _quadruped_stub(foot_z=[-0.03, -0.01], foot_vz=[0.0, 0.0], solver=solver)
    driver = DeformableTerrainDriver(cfg, art, num_envs=1, device="cpu")
    terrain = driver.terrain
    # Widen to 4 envs by hand so we can check selective resampling.
    terrain.k = torch.full((4, 1), 1000.0)
    terrain.mu = torch.full((4, 1), 0.5)
    env: Any = _FakeEnv(driver)

    torch.manual_seed(0)
    randomize_terrain_params(
        env, torch.tensor([1, 3]), ranges={"k": (50.0, 800.0), "mu": (0.4, 1.0)}
    )
    assert float(terrain.k[0, 0]) == 1000.0 and float(terrain.k[2, 0]) == 1000.0
    for i in (1, 3):
        assert 50.0 <= float(terrain.k[i, 0]) <= 800.0
        assert 0.4 <= float(terrain.mu[i, 0]) <= 1.0


def test_randomize_terrain_params_is_a_no_op_without_terrain() -> None:
    from genelab.mdp.events import randomize_terrain_params

    env: Any = _FakeEnv(None)
    randomize_terrain_params(env, torch.tensor([0]), ranges={"k": (1.0, 2.0)})  # no crash


def test_driver_injects_traction_from_foot_horizontal_velocity() -> None:
    solver = _FakeSolver()
    art = _quadruped_stub(
        foot_z=[-0.04, -0.04],
        foot_vz=[0.0, 0.0],
        solver=solver,
        foot_vxy=[(0.3, 0.0), (0.0, 0.0)],  # FL slides +x, FR stationary
    )
    cfg = DeformableTerrainCfg(
        k=1000.0, c=0.0, mu=0.5, surface_height=0.0, foot_link_names=("FL_foot", "FR_foot")
    )
    driver = DeformableTerrainDriver(cfg, art, num_envs=1, device="cpu")

    driver.apply()

    force, _ = solver.calls[0]
    # F_z = 1000 * 0.04 = 40 per foot; FL traction = -mu*F_z = -20 in x; FR no slip -> 0.
    assert float(force[0, 0, 0]) == pytest.approx(-20.0, rel=0.05)
    assert float(force[0, 1, 0]) == pytest.approx(0.0)


def test_driver_advances_residual_each_physics_step() -> None:
    solver = _FakeSolver()
    art = _quadruped_stub(foot_z=[-0.05, -0.05], foot_vz=[0.0, 0.0], solver=solver)
    cfg = DeformableTerrainCfg(
        k=1000.0,
        c=0.0,
        yield_depth=0.01,
        plastic_rate=5.0,
        recovery_time=1.0e9,
        foot_link_names=("FL_foot", "FR_foot"),
    )
    driver = DeformableTerrainDriver(cfg, art, num_envs=1, device="cpu", dt=0.1)
    driver.apply()
    driver.apply()
    # 2 steps * 5 * (0.05 - 0.01) * 0.1 = 0.04
    assert float(driver.terrain.state.residual.max()) == pytest.approx(0.04, rel=0.02)


def test_driver_stamps_the_footprint_map_at_foot_positions() -> None:
    solver = _FakeSolver()
    art = _quadruped_stub(foot_z=[-0.05, -0.05], foot_vz=[0.0, 0.0], solver=solver)
    cfg = DeformableTerrainCfg(
        k=1000.0,
        c=0.0,
        yield_depth=0.01,
        plastic_rate=5.0,
        recovery_time=1.0e9,
        footprint_map_size=(2.0, 2.0),
        footprint_map_resolution=0.1,
        foot_link_names=("FL_foot", "FR_foot"),
    )
    driver = DeformableTerrainDriver(cfg, art, num_envs=1, device="cpu", dt=0.1)
    driver.apply()
    assert driver.terrain.footprint_map is not None
    # feet sit at world (0, 0) in the stub -> the origin cell is stamped
    assert float(driver.terrain.footprint_map.read(torch.tensor([[[0.0, 0.0]]]))) > 0.0


def test_driver_reset_clears_terrain_state() -> None:
    solver = _FakeSolver()
    art = _quadruped_stub(foot_z=[-0.03, -0.01], foot_vz=[0.0, 0.0], solver=solver)
    cfg = DeformableTerrainCfg(foot_link_names=("FL_foot", "FR_foot"))
    driver = DeformableTerrainDriver(cfg, art, num_envs=1, device="cpu")
    driver.apply()
    driver.reset(torch.tensor([0]))
    assert float(driver.terrain.state.depth.abs().sum()) == 0.0


# ------------------------------------------------- privileged observation (fake env)


class _FakeEnv:
    def __init__(self, driver: DeformableTerrainDriver | None) -> None:
        self.deformable_terrain = driver


def test_terrain_sinkage_observation_exposes_privileged_per_foot_depth() -> None:
    from genelab.mdp.observations import terrain_sinkage

    solver = _FakeSolver()
    art = _quadruped_stub(foot_z=[-0.03, -0.01], foot_vz=[0.0, 0.0], solver=solver)
    cfg = DeformableTerrainCfg(
        k=1000.0, c=0.0, surface_height=0.0, foot_link_names=("FL_foot", "FR_foot")
    )
    driver = DeformableTerrainDriver(cfg, art, num_envs=1, device="cpu")
    driver.apply()
    env: Any = _FakeEnv(driver)

    obs = terrain_sinkage(env)
    assert torch.allclose(obs[0], torch.tensor([0.03, 0.01]))


def test_terrain_sinkage_observation_requires_configured_terrain() -> None:
    from genelab.mdp.observations import terrain_sinkage

    env: Any = _FakeEnv(None)
    with pytest.raises(RuntimeError):
        terrain_sinkage(env)


# --------------------------------------------------- end-to-end (Genesis runtime)


def test_analytic_support_settles_a_floorless_body_at_predicted_sinkage(
    genesis_runtime: Any,
) -> None:
    """With no rigid floor, the injected analytic F_z is the body's only support and
    settles it at ``z = -W/k`` below the virtual surface (the Spike-B2 result driven
    end to end through ``DeformableTerrain``)."""
    gs = genesis_runtime
    scene = gs.Scene(show_viewer=False)
    body = scene.add_entity(gs.morphs.Box(pos=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)))
    scene.build(n_envs=1)

    mass = float(body.get_mass())
    weight = mass * 9.81
    k = 1000.0
    cfg = DeformableTerrainCfg(k=k, c=2.0 * (k * mass) ** 0.5, surface_height=0.0)
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    link = body.links[0]

    for _ in range(800):
        terrain.apply_foot_forces(
            link.solver,
            foot_link_indices=(link.idx,),
            foot_height=body.get_pos()[:, 2:3],
            foot_vel_z=body.get_vel()[:, 2:3],
        )
        scene.step()

    z_end = float(body.get_pos()[0, 2])
    assert z_end == pytest.approx(-weight / k, abs=5e-3)


def test_traction_arrests_horizontal_sliding_on_soft_terrain(genesis_runtime: Any) -> None:
    """With mu>0 the analytic terrain grips: a body kicked sideways while supported has its
    slide arrested by Coulomb traction (the stage-1 ingredient that enables walking)."""
    gs = genesis_runtime
    scene = gs.Scene(show_viewer=False)
    body = scene.add_entity(gs.morphs.Box(pos=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)))
    scene.build(n_envs=1)

    mass = float(body.get_mass())
    k = 2000.0
    cfg = DeformableTerrainCfg(k=k, c=2.0 * (k * mass) ** 0.5, mu=1.0, surface_height=0.0)
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    link = body.links[0]

    def support() -> None:
        pos, vel = body.get_pos(), body.get_vel()
        terrain.apply_foot_forces(
            link.solver,
            foot_link_indices=(link.idx,),
            foot_height=pos[:, 2:3],
            foot_vel_z=vel[:, 2:3],
            foot_vel_xy=vel[:, :2].unsqueeze(1),
        )

    for _ in range(150):  # settle vertically
        support()
        scene.step()
    body.set_dofs_velocity(torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))  # kick +x
    v0 = float(body.get_vel()[0, :2].norm())
    for _ in range(150):  # traction acts
        support()
        scene.step()
    v1 = float(body.get_vel()[0, :2].norm())

    assert v0 > 0.8  # the kick took effect
    assert v1 < 0.2 * v0  # traction arrested most of the slide


def test_plastic_residual_deepens_the_footprint_under_sustained_load(
    genesis_runtime: Any,
) -> None:
    """Terrain with memory: a body held on plastic soft terrain keeps sinking as the
    footprint (residual) accumulates, instead of resting at a fixed equilibrium."""
    gs = genesis_runtime
    scene = gs.Scene(show_viewer=False)
    body = scene.add_entity(gs.morphs.Box(pos=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)))
    scene.build(n_envs=1)

    mass = float(body.get_mass())
    k = 4000.0
    weight = mass * 9.81
    cfg = DeformableTerrainCfg(
        k=k,
        c=2.0 * (k * mass) ** 0.5,
        surface_height=0.0,
        yield_depth=0.5 * weight / k,  # elastic equilibrium W/k sits past yield -> plastic
        plastic_rate=2.0,
        recovery_time=1.0e9,
    )
    terrain = DeformableTerrain(cfg, num_envs=1, num_feet=1)
    link = body.links[0]
    dt = float(scene.sim_options.dt)

    def support() -> None:
        pos, vel = body.get_pos(), body.get_vel()
        terrain.apply_foot_forces(
            link.solver,
            foot_link_indices=(link.idx,),
            foot_height=pos[:, 2:3],
            foot_vel_z=vel[:, 2:3],
            dt=dt,
        )

    for _ in range(50):  # initial settle
        support()
        scene.step()
    z_early = float(body.get_pos()[0, 2])
    for _ in range(400):  # residual keeps building -> footprint deepens
        support()
        scene.step()
    z_late = float(body.get_pos()[0, 2])

    assert float(terrain.state.residual.max()) > 0.0  # a footprint formed
    assert z_late < z_early - 1e-2  # and the body sank deeper into it
