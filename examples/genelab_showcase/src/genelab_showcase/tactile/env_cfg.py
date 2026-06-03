"""Tactile showcase env: a flat tactile plate pressing and dragging across dynamic shapes.

A flat square "plate" (the only articulation — a vertical ``press_z`` slide joint plus a
horizontal ``slide_x`` joint, bundled as ``tactile_plate.xml``) carries a dense grid of SDF
tactile probes on its underside. The runner eases it down onto three dynamic, equal-height
shapes (two balls and a cube), holds them under a pulsing press, and sweeps it sideways — so the
balls roll and the cube slides under the plate. :class:`~genelab.sensor.KinematicDepthSensor`
reports per-probe penetration depth, which the runner renders as a live 2-D pressure heatmap:
because the contact is compliant, the imprints emerge progressively, brighten with press force,
and track sideways as the shapes move. Nothing is written to disk.

This is geometric / SDF tactile (penetration depth across a fixed probe array), not a
deformable-skin FEM or optical tactile model — the plate is rigid and the "image" is the SDF
penetration sampled per probe, not a deformed gel surface.
"""

from pathlib import Path

from genelab import mdp
from genelab.actuator import ImplicitPDActuatorCfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.entity import ArticulationCfg, RigidObjectCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import EventTermCfg, TerminationTermCfg
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.sensor import KinematicDepthSensorCfg

# The bundled flat-plate articulation (vertical ``press_z`` + horizontal ``slide_x`` joints).
PLATE_MJCF: str = str(Path(__file__).with_name("tactile_plate.xml"))
PLATE_LINK: str = "plate"
PRESS_JOINT: str = "press_z"
DRAG_JOINT: str = "slide_x"

# Dense probe array on the plate underside. ``GRID_N × GRID_N`` probes span the plate footprint
# in the link-local frame; ``z = -0.012`` sits them ≈ 0.004 m proud of the plate's collision face
# so they dip into each shape's surface and read its height profile (a graded image, not a binary
# footprint). The rigid contact is compliant, so penetration depth — and thus the heatmap
# intensity — ramps up smoothly as the plate presses harder. Probe order is row-major in ``x``
# then ``y`` — reshape to ``(GRID_N, GRID_N)`` for the heatmap.
GRID_N: int = 32
_SPAN: float = 0.09
PAD_PROBES: tuple[tuple[float, float, float], ...] = tuple(
    (-_SPAN + 2 * _SPAN * i / (GRID_N - 1), -_SPAN + 2 * _SPAN * j / (GRID_N - 1), -0.012)
    for i in range(GRID_N)
    for j in range(GRID_N)
)
PROBE_RADIUS: float = 0.012


def _plate_robot_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        mjcf_path=PLATE_MJCF,
        init_pos=(0.0, 0.0, 0.0),
        # Both joints home at 0 → plate centred and hovering at its spawn height (0.13 m),
        # clear of the shapes.
        default_joint_pos={PRESS_JOINT: 0.0, DRAG_JOINT: 0.0},
        actuators={
            "plate": ImplicitPDActuatorCfg(
                target_names_expr=(rf"^({PRESS_JOINT}|{DRAG_JOINT})$",),
                stiffness=8000.0,
                damping=400.0,
                # High effort ceiling so the compliant contact keeps deepening as the plate
                # presses harder (heatmap intensity ramps with force) and so the horizontal drag
                # carries the shapes along instead of stalling at first contact.
                effort_limit=3000.0,
                velocity_limit=1.0,
                action_scale=0.18,
            ),
        },
    )


def tactile_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """Single-env flat tactile plate pressing — and dragging across — dynamic shapes."""

    # Dynamic (``fixed=False``) shapes resting on the ground with equal top height (z ≈ 0.08), so
    # the flat plate images all of them at once. High friction lets the plate's horizontal drag
    # carry them along: the spheres roll and the box slides under the plate, and their imprints
    # track across the heatmap. Spread across ``y`` so they don't collide as they move in ``x``.
    shapes: dict[str, ArticulationCfg | RigidObjectCfg] = {
        "ball_big": RigidObjectCfg(
            morph="sphere", size=(0.04,), init_pos=(-0.04, -0.05, 0.041), fixed=False, friction=1.2
        ),
        "ball_small": RigidObjectCfg(
            morph="sphere", size=(0.04,), init_pos=(-0.03, 0.05, 0.041), fixed=False, friction=1.2
        ),
        "cube": RigidObjectCfg(
            morph="box",
            size=(0.07, 0.07, 0.08),
            init_pos=(0.05, 0.0, 0.041),
            fixed=False,
            friction=1.2,
        ),
    }
    return ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=1,
            dt=0.01,
            substeps=2,
            steps=400,
            vis=False,
            gpu=True,
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(1.0, 1.0),
            entities=shapes,
            sensors=(
                KinematicDepthSensorCfg(
                    name="pad",
                    link_name=PLATE_LINK,
                    probe_local_pos=PAD_PROBES,
                    probe_radius=PROBE_RADIUS,
                ),
            ),
        ),
        decimation=2,
        episode_length_s=1000.0,
        device="cuda",
        robot=_plate_robot_cfg(),
        actions_cfg={
            "plate": JointPositionActionCfg(
                asset_name="robot",
                joint_names=(rf"^({PRESS_JOINT}|{DRAG_JOINT})$",),
                use_default_offset=True,
            ),
        },
        terminations_cfg={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        },
        events_cfg={
            "reset_joints": EventTermCfg(
                mode="reset",
                func=mdp.reset_joints_to_default,
                params={"pos_jitter": 0.0, "vel_jitter": 0.0},
            ),
        },
    )
