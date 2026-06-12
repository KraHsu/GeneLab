"""Three balls bouncing on a pneumatic air mattress — cross-foot chamber coupling demo.

The mattress is **one sealed air chamber** (:func:`genelab.terrains.pneumatic_normal_force`
as the terrain's ``normal_law``): the support under every contacting ball is the *same*
chamber pressure, an instantaneous function of the total volume all balls displace. Drop
one ball onto the mattress and the balls resting elsewhere get tossed into the air —
coupling that independent per-contact springs cannot produce.

There is no rigid contact with the floor (disjoint collision masks, as in the soft-terrain
envs): the analytic chamber force is the only thing holding the balls up. The mattress box
is a visual.

Run with the viewer:

    PYTHONPATH=examples/genelab_soft_terrain/src uv run python -m genelab_soft_terrain.air_mattress
"""

from functools import partial

import torch

from genelab.terrains import DeformableTerrain, DeformableTerrainCfg, pneumatic_normal_force

_TOP = 0.4  # mattress (undisturbed chamber) top surface, m
_RADIUS = 0.1
_CAPACITY = 0.35  # total displaced depth at chamber bottom-out, m
_PRESSURE_STIFFNESS = 600.0  # N per m of total displaced depth, at small fill
_DAMPING = 10.0
_MU = 0.3

# Two balls rest on the mattress; the third drops onto it from above.
_REST_POS = ((0.3, 0.0), (-0.15, 0.26))
_DROP_POS = (-0.15, -0.26)
_DROP_HEIGHT = 2.2


def run(steps: int = 600, show_viewer: bool = True) -> dict[str, float]:
    """Simulate the demo; returns coupling metrics (used by the gated smoke test)."""
    import genesis as gs

    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cpu, logging_level="warning")  # type: ignore[attr-defined]

    scene = gs.Scene(
        show_viewer=show_viewer,
        sim_options=gs.options.SimOptions(dt=0.005),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.2, -2.2, 1.6), camera_lookat=(0.0, 0.0, 0.5), max_FPS=60
        ),
    )
    # Floor renders but does not collide with the balls (disjoint masks): the chamber
    # force is the only support. Balls share a group so they collide with each other.
    scene.add_entity(gs.morphs.Plane(contype=0b01, conaffinity=0b01))
    scene.add_entity(
        gs.morphs.Box(pos=(0.0, 0.0, _TOP / 2), size=(1.6, 1.6, _TOP), collision=False),
        surface=gs.surfaces.Default(color=(0.55, 0.65, 0.95)),
    )
    colors = ((0.9, 0.3, 0.3), (0.3, 0.8, 0.4), (0.95, 0.8, 0.25))
    spawns = (
        (*_REST_POS[0], _TOP + _RADIUS),
        (*_REST_POS[1], _TOP + _RADIUS),
        (*_DROP_POS, _DROP_HEIGHT),
    )
    balls = [
        scene.add_entity(
            gs.morphs.Sphere(pos=pos, radius=_RADIUS, contype=0b10, conaffinity=0b10),
            surface=gs.surfaces.Default(color=color),
        )
        for pos, color in zip(spawns, colors, strict=True)
    ]
    scene.build(n_envs=1)

    links = [ball.links[0] for ball in balls]
    solver = links[0].solver
    link_indices = tuple(link.idx for link in links)

    terrain = DeformableTerrain(
        DeformableTerrainCfg(
            k=_PRESSURE_STIFFNESS,
            c=_DAMPING,
            mu=_MU,
            surface_height=_TOP,
            normal_law=partial(pneumatic_normal_force, capacity=_CAPACITY),
        ),
        num_envs=1,
        num_feet=len(balls),
    )

    drop_contacted = False
    rest_z_at_contact = 0.0
    toss = 0.0
    min_z = float("inf")
    for _ in range(steps):
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
        scene.step()

        rest_z = float(pos[:2, 2].max())
        min_z = min(min_z, float(pos[:, 2].min()))
        if not drop_contacted and float(bottom_z[0, 2]) < _TOP:
            drop_contacted = True
            rest_z_at_contact = rest_z
        if drop_contacted:
            toss = max(toss, rest_z - rest_z_at_contact)

    return {"toss": toss, "min_z": min_z, "rest_z_at_contact": rest_z_at_contact}


def main() -> None:
    metrics = run()
    print(
        f"resting balls tossed {metrics['toss'] * 100:.1f} cm by the drop "
        f"(lowest ball z {metrics['min_z']:.2f} m — never reached the floor)"
    )


if __name__ == "__main__":
    main()
