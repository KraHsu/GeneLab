"""Three balls bouncing on a pneumatic air mattress — cross-foot chamber coupling demo.

The mattress is **one sealed air chamber** (:func:`genelab.terrains.pneumatic_normal_force`
as the terrain's ``normal_law``): the support under every contacting ball is the *same*
chamber pressure, an instantaneous function of the total volume all balls displace. Drop
one ball onto the mattress and the balls resting elsewhere get tossed into the air —
coupling that independent per-contact springs cannot produce.

There is no rigid contact with the floor (non-colliding ground plane, as in the
soft-terrain envs): the analytic chamber force is the only thing holding the balls up.
The mattress box is a render-only visual. With the viewer, an ImGui panel exposes the
chamber (capacity / stiffness sliders, live), a fill readout, and a **Drop ball** button.

Registered as the play-only task ``Genelab-Air-Mattress-Demo-v0``:

    genelab play Genelab-Air-Mattress-Demo-v0
"""

import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch

from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.entity import RigidObjectCfg
from genelab.materials import SurfaceCfg
from genelab.scene import InteractiveScene
from genelab.terrains import DeformableTerrain, DeformableTerrainCfg, pneumatic_normal_force

_TOP = 0.4  # mattress (undisturbed chamber) top surface, m
_RADIUS = 0.1
_CAPACITY = 0.35  # total displaced depth at chamber bottom-out, m
_PRESSURE_STIFFNESS = 600.0  # N per m of total displaced depth, at small fill
_DAMPING = 10.0
_MU = 0.3
_DT = 0.005

# Two balls rest on the mattress; the third drops onto it from above.
_BALLS = ("ball_red", "ball_green", "ball_drop")
_REST_POS = ((0.3, 0.0), (-0.15, 0.26))
_DROP_POS = (-0.15, -0.26)
_DROP_HEIGHT = 2.2
_REDROP_EVERY = 1_000  # steps (~5 s): lift the dropper back up so the show loops


@dataclass
class AirMattressDemoCfg:
    """The air-mattress demo: simulation settings + scene + chamber model."""

    simulation: SimulationCfg = field(default_factory=SimulationCfg)
    scene: InteractiveSceneCfg = field(default_factory=InteractiveSceneCfg)
    terrain: DeformableTerrainCfg = field(default_factory=DeformableTerrainCfg)


def air_mattress_demo_cfg() -> AirMattressDemoCfg:
    """Three balls over one sealed chamber; the chamber force is the only support."""
    colors = ((0.9, 0.3, 0.3), (0.3, 0.8, 0.4), (0.95, 0.8, 0.25))
    spawns = (
        (*_REST_POS[0], _TOP + _RADIUS),
        (*_REST_POS[1], _TOP + _RADIUS),
        (*_DROP_POS, _DROP_HEIGHT),
    )
    entities: dict[str, Any] = {
        "mattress": RigidObjectCfg(
            morph="box",
            size=(1.6, 1.6, _TOP),
            init_pos=(0.0, 0.0, _TOP / 2),
            fixed=True,
            collision=False,  # render-only: the chamber model, not the box, is the support
            surface=SurfaceCfg(color=(0.55, 0.65, 0.95)),
        )
    }
    for name, pos, color in zip(_BALLS, spawns, colors, strict=True):
        entities[name] = RigidObjectCfg(
            morph="sphere",
            size=(_RADIUS,),
            init_pos=pos,
            fixed=False,
            surface=SurfaceCfg(color=color),
        )
    return AirMattressDemoCfg(
        simulation=SimulationCfg(
            num_envs=1,
            dt=_DT,
            steps=600,
            vis=True,
            gpu=False,  # three balls: CPU is plenty and never contends for the GPU
            camera_pos=(2.2, -2.2, 1.6),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        scene=InteractiveSceneCfg(ground_plane_collision=False, entities=entities),
        terrain=DeformableTerrainCfg(
            k=_PRESSURE_STIFFNESS,
            c=_DAMPING,
            mu=_MU,
            surface_height=_TOP,
            normal_law=partial(pneumatic_normal_force, capacity=_CAPACITY),
        ),
    )


@dataclass
class _Controls:
    """State shared between the ImGui panel (viewer thread) and the step loop."""

    stiffness: float = _PRESSURE_STIFFNESS
    capacity: float = _CAPACITY
    fill: float = 0.0  # readout: displaced / capacity, written by the loop
    drop: bool = False


def _chamber_panel(controls: _Controls) -> Any:
    """``callback(imgui)`` for :attr:`SimulationCfg.panels`: chamber knobs + Drop."""

    def draw(imgui: Any) -> None:
        imgui.separator_text("Air chamber")
        _, controls.stiffness = imgui.slider_float(
            "stiffness", float(controls.stiffness), 100.0, 3000.0
        )
        _, controls.capacity = imgui.slider_float("capacity", float(controls.capacity), 0.1, 0.6)
        imgui.text(f"chamber fill: {controls.fill * 100.0:5.1f} %")
        if imgui.button("Drop ball"):
            controls.drop = True

    return draw


def run(
    cfg: AirMattressDemoCfg | None = None,
    *,
    steps: int | None = None,
    show_viewer: bool | None = None,
    redrop_every: int | None = None,
    realtime: bool = False,
) -> dict[str, float]:
    """Build and step the demo; returns coupling metrics (used by the gated smoke test).

    With the viewer, the panel's sliders drive the chamber live (stiffness through the
    per-env ``k`` tensor, capacity through a closure normal law) and **Drop ball** lifts
    the dropper back to its spawn; ``redrop_every`` does the same on a timer.
    """
    cfg = cfg or air_mattress_demo_cfg()
    sim = cfg.simulation
    if show_viewer is not None:
        sim.vis = show_viewer
    bound = steps if steps is not None else int(sim.steps)

    controls = _Controls(stiffness=float(cfg.terrain.k))
    law = cfg.terrain.normal_law
    if isinstance(law, partial) and "capacity" in law.keywords:
        controls.capacity = float(law.keywords["capacity"])
    if sim.vis:
        sim.panels = [*sim.panels, _chamber_panel(controls)]

    scene = InteractiveScene(sim, cfg.scene, device_hint="cpu")
    scene.build()
    try:
        return _step_loop(scene, cfg, controls, bound, redrop_every, realtime)
    finally:
        scene.close()


def _step_loop(
    scene: InteractiveScene,
    cfg: AirMattressDemoCfg,
    controls: _Controls,
    bound: int,
    redrop_every: int | None,
    realtime: bool,
) -> dict[str, float]:
    balls = [scene[name].gs_handle for name in _BALLS]
    links = [ball.links[0] for ball in balls]
    solver = links[0].solver
    link_indices = tuple(link.idx for link in links)

    # The panel's capacity slider reaches the physics through this closure (the per-env
    # ``k`` tensor below carries the stiffness slider).
    cfg.terrain.normal_law = lambda d, dr, k, c: pneumatic_normal_force(
        d, dr, k, c, capacity=controls.capacity
    )
    terrain = DeformableTerrain(cfg.terrain, num_envs=1, num_feet=len(balls))

    drop_contacted = False
    rest_z_at_contact = 0.0
    toss = 0.0
    min_z = float("inf")
    for step in range(bound):
        timer_drop = bool(redrop_every) and step > 0 and step % int(redrop_every or 1) == 0
        if controls.drop or timer_drop:
            controls.drop = False
            balls[2].set_pos(torch.tensor([[*_DROP_POS, _DROP_HEIGHT]]), zero_velocity=True)
        terrain.k.fill_(controls.stiffness)

        pos = torch.stack([torch.as_tensor(b.get_pos()).reshape(-1) for b in balls], dim=0)
        vel = torch.stack([torch.as_tensor(b.get_vel()).reshape(-1) for b in balls], dim=0)
        # The balls are the terrain's "feet": contact point is the ball's bottom.
        bottom_z = (pos[:, 2] - _RADIUS).unsqueeze(0)
        terrain.apply_foot_forces(
            solver,
            link_indices,
            foot_height=bottom_z,
            foot_vel_z=vel[:, 2].unsqueeze(0),
            foot_vel_xy=vel[:, :2].unsqueeze(0),
        )
        controls.fill = float(terrain.state.depth.clamp_min(0.0).sum() / controls.capacity)
        scene.step()
        if scene.viewer_closed:
            break
        if realtime:
            time.sleep(_DT)

        rest_z = float(pos[:2, 2].max())
        min_z = min(min_z, float(pos[:, 2].min()))
        if not drop_contacted and float(bottom_z[0, 2]) < cfg.terrain.surface_height:
            drop_contacted = True
            rest_z_at_contact = rest_z
        if drop_contacted:
            toss = max(toss, rest_z - rest_z_at_contact)

    return {"toss": toss, "min_z": min_z, "rest_z_at_contact": rest_z_at_contact}


def main() -> None:
    # ~60 s of show, the ball re-dropping every ~5 s, paced to real time.
    metrics = run(steps=12_000, show_viewer=True, redrop_every=_REDROP_EVERY, realtime=True)
    print(
        f"resting balls tossed {metrics['toss'] * 100:.1f} cm by the drop "
        f"(lowest ball z {metrics['min_z']:.2f} m — never reached the floor)"
    )


if __name__ == "__main__":
    main()
