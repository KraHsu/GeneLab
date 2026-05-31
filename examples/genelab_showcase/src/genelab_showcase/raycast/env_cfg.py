"""RayCast showcase env: Franka with Genesis-backed mesh ray-casters scanning real props.

Unlike the analytic :class:`~genelab.sensor.RayCastSensor` (ray-vs-ground / heightfield),
:class:`~genelab.sensor.MeshRayCastSensor` casts against the BVH of real scene meshes, so the
rays hit the boxes and spheres placed in front of the robot. Two patterns scan along the
gripper-forward axis (the Franka ``hand`` link's +z); a slow joint-1 sweep drags the scan
across the props. The runner renders the hit point cloud in a separate window — nothing is
drawn in the main scene and nothing is written to disk.
"""

from genelab import mdp
from genelab.asset_zoo import FrankaPandaCfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.entity import ArticulationCfg, RigidObjectCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import EventTermCfg, TerminationTermCfg
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.sensor import MeshGridPattern, MeshRayCastSensorCfg, MeshSphericalPattern

# Gripper-forward = the Franka ``hand`` link's +z (the same axis the #1 wrist camera looks
# along). Grid ray directions are in the sensor/link frame, so +z points the scan forward.
_FORWARD = (0.0, 0.0, 1.0)


def raycast_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """Single-env Franka with a grid + a spherical-fan mesh ray-caster scanning front props."""

    robot_cfg = FrankaPandaCfg()
    # Props in front of the robot (the gripper-forward footprint, ~x 0.4-0.5), raised so the
    # rays read varying distances. ``use_visual_raycasting`` is harmless for these solid
    # bodies (their collision faces are in the BVH anyway) but keeps the intent explicit.
    props: dict[str, ArticulationCfg | RigidObjectCfg] = {
        "box_tall": RigidObjectCfg(
            morph="box",
            size=(0.08, 0.08, 0.30),
            init_pos=(0.45, 0.0, 0.15),
            use_visual_raycasting=True,
        ),
        "box_wide": RigidObjectCfg(
            morph="box",
            size=(0.12, 0.12, 0.16),
            init_pos=(0.40, 0.17, 0.08),
            use_visual_raycasting=True,
        ),
        "box_low": RigidObjectCfg(
            morph="box",
            size=(0.10, 0.10, 0.10),
            init_pos=(0.52, -0.18, 0.05),
            use_visual_raycasting=True,
        ),
        "ball_a": RigidObjectCfg(
            morph="sphere", size=(0.07,), init_pos=(0.50, 0.10, 0.07), use_visual_raycasting=True
        ),
        "ball_b": RigidObjectCfg(
            morph="sphere", size=(0.05,), init_pos=(0.38, -0.08, 0.05), use_visual_raycasting=True
        ),
    }

    return ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(num_envs=1, dt=0.01, substeps=2, steps=200, vis=False, gpu=True),
        scene=InteractiveSceneCfg(
            env_spacing=(2.0, 2.0),
            entities=props,
            sensors=(
                MeshRayCastSensorCfg(
                    name="mesh_grid",
                    link_name="hand",
                    pattern=MeshGridPattern(resolution=0.03, size=(0.18, 0.18), direction=_FORWARD),
                    max_range=2.0,
                    # Visualisation is the separate point-cloud window only; no in-scene spheres.
                ),
                MeshRayCastSensorCfg(
                    name="mesh_cone",
                    link_name="hand",
                    pattern=MeshSphericalPattern(fov=(60.0, 60.0), n_points=(10, 10)),
                    # The spherical fan centers on the sensor's local +x; a -90° pitch maps +x
                    # onto the hand's +z (gripper-forward) so the fan scans the props ahead.
                    euler_offset=(0.0, -90.0, 0.0),
                    # Push the fan origin ~12 cm forward (along the gripper) so it clears the
                    # finger tips and the rays actually reach the props instead of the gripper.
                    pos_offset=(0.0, 0.0, 0.12),
                    max_range=2.5,
                ),
            ),
        ),
        decimation=2,
        episode_length_s=20.0,
        device="cuda",
        robot=robot_cfg,
        actions_cfg={
            "panda_arm": JointPositionActionCfg(
                asset_name="robot", joint_names=(r"^joint[1-7]$",), use_default_offset=True
            ),
            "panda_hand": JointPositionActionCfg(
                asset_name="robot", joint_names=(r"finger_joint.*",), use_default_offset=True
            ),
        },
        terminations_cfg={"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)},
        events_cfg={
            "reset_joints": EventTermCfg(
                mode="reset",
                func=mdp.reset_joints_to_default,
                params={"pos_jitter": 0.0, "vel_jitter": 0.0},
            ),
        },
    )
