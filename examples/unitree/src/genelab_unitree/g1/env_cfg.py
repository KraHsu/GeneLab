"""Velocity-tracking env config for the Unitree G1 on flat ground.

1:1 mirror of the reference's ``unitree_g1_flat_env_cfg`` (``tasks.velocity.config.g1``),
adapted to GeneLab's slim manager surface. Every reward / event / curriculum from
The reference is wired through the GeneLab core abstractions landed in
phases P1–P5 of the parity work:

* P1 — locomotion gait rewards (``feet_clearance`` / ``feet_swing_height`` /
  ``feet_slip`` / ``soft_landing`` / ``body_angular_velocity_penalty``),
  ``commands_vel`` curriculum, ``reset_joints_by_offset`` event,
  ``ContactSensor.first_contact``.
* P2 — env-group fields on ``UniformVelocityCommand`` (``rel_heading_envs``,
  ``rel_forward_envs``).
* P3 — multi-frame ``TerrainHeightSensor`` for ``foot_height_scan``,
  ``SelfContactSensor`` for self-collision, ``RootAngularMomentumSensor``,
  ``ContactSensor.force_history``.
* P4 — startup DR events (``geom_friction`` / ``encoder_bias`` /
  ``body_com_offset``).
* P5 — diagnostic ``MetricsManager`` (``mean_action_acc``), ``SceneEntityCfg``
  used uniformly for every link/joint selector.

Knob values match the reference exactly except for the simulation timestep — GeneLab
keeps ``dt=0.002, decimation=10`` (Genesis solver) instead of the reference's
``dt=0.005, decimation=4`` (MuJoCo). Control frequency stays at 50 Hz so the
policy sees the same control cadence.
"""

import math
from typing import TYPE_CHECKING

from genelab import mdp
from genelab.asset_zoo.unitree_g1 import UnitreeG1Cfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    MetricsTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.mdp.commands.velocity_command import UniformVelocityCommandCfg
from genelab.mdp.noise import Unoise
from genelab.sensor import (
    BodyVelocitySensorCfg,
    ContactSensorCfg,
    GridPattern,
    RootAngularMomentumSensorCfg,
    SelfContactSensorCfg,
    SensorCfg,
    TerrainHeightSensorCfg,
)

if TYPE_CHECKING:
    from genelab.terrains import TerrainGeneratorCfg

# IMU site offset from the pelvis link origin; matches g1.xml's <site name="imu_in_pelvis">.
_IMU_OFFSET = (0.04525, 0.0, -0.08339)

# Foot-site offset from each ankle_roll_link origin; matches g1.xml's
# <site name="left_foot"/"right_foot" pos="0.04 0 -0.037">. The reference evaluates
# feet_clearance / feet_swing_height / feet_slip and foot_height_scan at the
# site, not at the link origin — sitting ~0.037 m lower matters next to the
# 0.1 m target swing height.
_G1_FOOT_SITE_OFFSET: tuple[float, float, float] = (0.04, 0.0, -0.037)

# Link names — Genesis-side equivalents of the body names the reference uses.
_TORSO_LINK = "torso_link"
_PELVIS_LINK = "pelvis"
_G1_FOOT_LINKS: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")

# Per-joint posture standard deviations used by the ``pose`` reward. Smaller std =
# tighter tolerance. Lower body looser at running speed; arms/wrists looser to allow
# natural swing. Mirrors the reference's G1 std table verbatim.
_G1_POSE_STD_WALKING: dict[str, float] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
}

_G1_POSE_STD_RUNNING: dict[str, float] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
}


def _feet_cfg() -> SceneEntityCfg:
    """Reusable selector pointing at the two foot links + the reference foot site.

    The ``link_offsets`` make feet_clearance / feet_swing_height / feet_slip
    evaluate at the ``left_foot``/``right_foot`` *site* (G1 XML `g1.xml:100,149`),
    matching the reference's reward signal.
    """
    return SceneEntityCfg(
        name="robot",
        link_names=_G1_FOOT_LINKS,
        link_offsets=(_G1_FOOT_SITE_OFFSET,) * len(_G1_FOOT_LINKS),
    )


def _single_play_spawn_xy(terrain: "TerrainGeneratorCfg | None") -> tuple[float, float]:
    """World (x, y) the lone play robot spawns at — used only to frame the viewer camera.

    Flat ground spawns at the origin. On a height-field it spawns at one terrain cell: row 0,
    column chosen deterministically from ``cfg.seed`` by ``TerrainImporter.init_per_env_state``
    (replicated here so the camera can be framed before the scene is built). The level
    curriculum may step the robot one row downrange after the first reset, so the camera is
    placed far enough back to keep it in view.
    """
    if terrain is None:
        return 0.0, 0.0
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(terrain.seed + 1)
    col = int(torch.randint(0, terrain.num_cols, (1,), generator=gen, device="cpu").item())
    px, py, _ = terrain.pos
    sx, sy = terrain.subterrain_size
    return px + 0.5 * sx, py + (col + 0.5) * sy


def _obs_terms() -> dict[str, ObservationTermCfg]:
    """Actor / critic-shared observation terms.

    Noise levels match the reference 1:1. The ``joint_vel`` term has no extra
    ``scale`` factor (reference parity) — the policy reads raw rad/s + noise.
    """
    # Order matches ``tasks.velocity.velocity_env_cfg.make_velocity_env_cfg``'s
    # actor_terms dict — important for the policy net's input concat layout when
    # comparing training curves across the two backends.
    return {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.sensor_data,
            params={"sensor_name": "imu_lin_vel"},
            noise=Unoise(-0.5, 0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.sensor_data,
            params={"sensor_name": "imu_ang_vel"},
            noise=Unoise(-0.2, 0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity, noise=Unoise(-0.05, 0.05)
        ),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(-0.01, 0.01)),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(-1.5, 1.5)),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "velocity_commands": ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "twist"}
        ),
    }


def _policy_obs_group() -> ObservationGroupCfg:
    return ObservationGroupCfg(enable_corruption=True, terms=_obs_terms())


def _critic_obs_group() -> ObservationGroupCfg:
    terms = _obs_terms()
    # Privileged state: critic sees the foot kinematics + clean ground clearance.
    terms["foot_height"] = ObservationTermCfg(
        func=mdp.sensor_data, params={"sensor_name": "foot_height_scan"}
    )
    terms["foot_air_time"] = ObservationTermCfg(
        func=mdp.foot_air_time, params={"sensor_name": "feet_ground_contact"}
    )
    terms["foot_contact"] = ObservationTermCfg(
        func=mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}
    )
    terms["foot_contact_forces"] = ObservationTermCfg(
        func=mdp.foot_contact_forces, params={"sensor_name": "feet_ground_contact"}
    )
    return ObservationGroupCfg(enable_corruption=False, terms=terms)


def _command_vel_stages() -> list[dict]:
    """Three-stage command-range expansion driven by ``env.common_step_counter``.

    Matches the reference's G1 cfg verbatim: starts at modest range, widens after 5k
    iterations × 24 steps and again at 10k × 24. The schedule scaffolds the
    policy from "track easy commands" up to "track aggressive ones" without
    blowing the early-training rewards.
    """
    return [
        {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
        {"step": 5_000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
        {"step": 10_000 * 24, "lin_vel_x": (-2.0, 3.0)},
    ]


def _velocity_env_cfg_base(
    *,
    play: bool = False,
    terrain: "TerrainGeneratorCfg | None" = None,
    extra_sensors: tuple[SensorCfg, ...] = (),
    extra_actor_obs: dict[str, ObservationTermCfg] | None = None,
    extra_critic_obs: dict[str, ObservationTermCfg] | None = None,
    extra_curriculum: dict[str, CurriculumTermCfg] | None = None,
) -> ManagerBasedRlEnvCfg:
    """Shared velocity-tracking skeleton for the Unitree G1.

    Both ``unitree_g1_velocity_env_cfg`` (flat) and
    ``unitree_g1_velocity_rough_env_cfg`` (rough) build on this. The flat task
    passes no deltas; the rough task injects a heightfield ``terrain``, a
    ``height_scan`` sensor + observation, and the ``terrain_levels`` curriculum.
    Keeping the body here means the two configs can never drift apart on the
    rewards / events / DR they share.
    """
    robot_entity_cfg = UnitreeG1Cfg()

    policy_obs = _policy_obs_group()
    policy_obs.terms.update(extra_actor_obs or {})
    critic_obs = _critic_obs_group()
    critic_obs.terms.update(extra_critic_obs or {})

    cfg = ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            # Play is an interactive single-robot teleop session driven by the ImGui twist
            # sliders, so it builds exactly one G1. (Pass ``--num-envs N`` to fan out more.)
            num_envs=4096 if not play else 1,
            dt=0.002,
            substeps=1,
            vis=play,
            # Run Genesis on the GPU. ``SimulationCfg.gpu`` defaults to False (CPU
            # backend); without this the G1 humanoid physics runs entirely on CPU —
            # GPU idle, ~30x slower per step (~1480 ms vs ~50 ms / step @1024 envs).
            gpu=True,
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(2.5, 2.5),
            terrain=terrain,
            sensors=(
                BodyVelocitySensorCfg(
                    name="imu_lin_vel",
                    link_name=_PELVIS_LINK,
                    offset=_IMU_OFFSET,
                    measure="lin_vel",
                ),
                BodyVelocitySensorCfg(
                    name="imu_ang_vel",
                    link_name=_PELVIS_LINK,
                    offset=_IMU_OFFSET,
                    measure="ang_vel",
                ),
                # reference parity: per-foot contact for slip / clearance / first-contact rewards.
                # ``history_length`` is left at 0 because the velocity task doesn't sample the
                # contact-force history window — ``self_collision`` (below) owns that signal.
                ContactSensorCfg(
                    name="feet_ground_contact",
                    link_names=_G1_FOOT_LINKS,
                    track_air_time=True,
                ),
                # ``foot_height_scan`` (reference parity): single down-pointing ray per foot,
                # ``reduction="min"`` so a multi-ray pattern stays a no-op on flat ground.
                # On a heightfield, the inner ``RayCastSensor`` walks the terrain mesh.
                # ``link_offsets`` anchors the ray at the ``left_foot``/``right_foot`` site
                # (g1.xml:100,149) instead of the ankle_roll_link origin — flat-ground
                # height becomes ``link_z + offset_z`` so the 0.037 m site z-offset
                # matters when comparing to ``target_height=0.1``.
                TerrainHeightSensorCfg(
                    name="foot_height_scan",
                    link_names=_G1_FOOT_LINKS,
                    pattern=GridPattern(resolution=1.0, size=(0.0, 0.0)),
                    reduction="min",
                    link_offsets=(_G1_FOOT_SITE_OFFSET,) * len(_G1_FOOT_LINKS),
                ),
                # the reference's ``self_collision`` ContactSensor used a subtree-vs-subtree filter on
                # the pelvis subtree; in GeneLab the SelfContactSensor reads Genesis's pair
                # list and reports a per-env scalar (sum of |F| across self-contact pairs).
                # ``history_length=4`` matches the reference so the cost counts hits in a 4-step window.
                SelfContactSensorCfg(
                    name="self_collision",
                    force_threshold=10.0,
                    history_length=4,
                ),
                # Whole-body angular momentum for the ``angular_momentum`` penalty. Orbital
                # approximation (Σ m·r×v) — see ``sensor/angular_momentum.py``.
                RootAngularMomentumSensorCfg(name="root_angmom"),
                *extra_sensors,
            ),
        ),
        decimation=10,
        episode_length_s=20.0,
        device="cuda",
        robot=robot_entity_cfg,
        actions_cfg={
            # ``scale=None`` inherits the per-actuator-group ``0.25 * effort/stiffness`` set in
            # ``asset_zoo/unitree_g1.py`` — same formula as the reference's ``G1_ACTION_SCALE`` dict.
            "joint_pos": JointPositionActionCfg(
                asset_name="robot",
                joint_names=(".*",),
                use_default_offset=True,
            )
        },
        observations_cfg={
            "policy": policy_obs,
            "critic": critic_obs,
        },
        commands_cfg={
            "twist": UniformVelocityCommandCfg(
                asset_name="robot",
                resampling_time_range=(3.0, 8.0),
                # Direct yaw-rate twist: the third command component is a *directly commanded*
                # ωz (vw) the policy tracks, not a heading target. ``heading_command=False`` (and
                # no heading group) means the 70% of non-standing / non-forward envs sample a raw
                # ωz, so the policy learns yaw-rate tracking; teleop's wz slider then drives yaw
                # directly instead of being overwritten by a heading PD. Groups: 10% standing,
                # 20% strict forward (vx≥0.3, ωz=0), 70% free (vx, vy, ωz all sampled).
                rel_standing_envs=0.1,
                rel_heading_envs=0.0,
                rel_forward_envs=0.2,
                heading_command=False,
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(-1.0, 1.0),
                    lin_vel_y=(-1.0, 1.0),
                    ang_vel_z=(-0.5, 0.5),
                    heading=(-math.pi, math.pi),
                ),
            )
        },
        rewards_cfg={
            # --- velocity tracking (positive) ---
            "track_lin_vel": RewardTermCfg(
                func=mdp.track_linear_velocity_xy_exp,
                weight=2.0,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            ),
            "track_ang_vel": RewardTermCfg(
                func=mdp.track_angular_velocity_z_exp,
                weight=2.0,
                params={"command_name": "twist", "std": math.sqrt(0.5)},
            ),
            "upright": RewardTermCfg(
                func=mdp.upright_exp,
                weight=1.0,
                # the reference targets ``torso_link`` (not pelvis) so the reward penalises
                # torso tilt — the part of the body humans "read" as posture — rather
                # than the floating base. When the waist joints flex, the two diverge.
                params={
                    "std": math.sqrt(0.2),
                    "asset_cfg": SceneEntityCfg(name="robot", link_names=(_TORSO_LINK,)),
                },
            ),
            "pose": RewardTermCfg(
                func=mdp.variable_posture,
                weight=1.0,
                params={
                    "command_name": "twist",
                    "walking_threshold": 0.05,
                    "running_threshold": 1.5,
                    "default_std": 0.3,
                    "std_standing": {".*": 0.05},
                    "std_walking": _G1_POSE_STD_WALKING,
                    "std_running": _G1_POSE_STD_RUNNING,
                },
            ),
            # --- gait shaping (penalties, weight < 0) ---
            "body_ang_vel": RewardTermCfg(
                func=mdp.body_angular_velocity_penalty,
                weight=-0.05,
                params={"asset_cfg": SceneEntityCfg(name="robot", link_names=(_TORSO_LINK,))},
            ),
            "angular_momentum": RewardTermCfg(
                func=mdp.angular_momentum_penalty,
                weight=-0.02,
                params={"sensor_name": "root_angmom"},
            ),
            "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
            "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
            # Diverges from reference parity: 2026-05-17 training showed gait collapse
            # (peak swing ≈2.4cm, air_time ≈55ms), so we turn ``air_time`` on for
            # positive stepping signal and quadruple ``foot_swing_height`` so the
            # peak-vs-target landing penalty actually drives the policy. See the
            # 2026-05-17_09-11-25 run for the baseline metrics.
            "air_time": RewardTermCfg(
                func=mdp.feet_air_time,
                weight=1.0,
                params={
                    "sensor_name": "feet_ground_contact",
                    "threshold_min": 0.05,
                    "threshold_max": 0.5,
                    "command_name": "twist",
                    "command_threshold": 0.5,
                },
            ),
            "foot_clearance": RewardTermCfg(
                func=mdp.feet_clearance,
                weight=-2.0,
                params={
                    "asset_cfg": _feet_cfg(),
                    "target_height": 0.1,
                    "command_name": "twist",
                    "command_threshold": 0.05,
                    # Multi-frame ``foot_height_scan`` returns (B, 2) — one per foot, in
                    # ``_G1_FOOT_LINKS`` order. On flat ground this equals link-z; the same
                    # call site picks up height-field clearance on rough terrain.
                    "height_sensor_name": "foot_height_scan",
                },
            ),
            "foot_swing_height": RewardTermCfg(
                func=mdp.feet_swing_height,
                weight=-1.0,
                params={
                    "sensor_name": "feet_ground_contact",
                    "asset_cfg": _feet_cfg(),
                    "target_height": 0.1,
                    "command_name": "twist",
                    "command_threshold": 0.05,
                },
            ),
            "foot_slip": RewardTermCfg(
                func=mdp.feet_slip,
                weight=-0.1,
                params={
                    "sensor_name": "feet_ground_contact",
                    "asset_cfg": _feet_cfg(),
                    "command_name": "twist",
                    "command_threshold": 0.05,
                },
            ),
            "soft_landing": RewardTermCfg(
                func=mdp.soft_landing,
                weight=-1e-5,
                params={
                    "sensor_name": "feet_ground_contact",
                    "command_name": "twist",
                    "command_threshold": 0.05,
                },
            ),
            "self_collisions": RewardTermCfg(
                func=mdp.self_collision_cost,
                weight=-1.0,
                # Force threshold lives on ``SelfContactSensorCfg.force_threshold`` (10.0
                # above); the reward used to accept a redundant param of the same name,
                # which is dropped here so the cfg has one source of truth.
                params={"sensor_name": "self_collision"},
            ),
        },
        terminations_cfg={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "fell_over": TerminationTermCfg(
                func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
            ),
        },
        events_cfg={
            "reset_base": EventTermCfg(
                mode="reset",
                func=mdp.reset_root_state_uniform,
                params={
                    "pose_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (0.01, 0.05),
                        "yaw": (-math.pi, math.pi),
                    },
                    "velocity_range": {},
                },
            ),
            # the reference uses ``reset_joints_by_offset`` with (0, 0) ranges — effectively default
            # pose, but the function is the same call site as the offset variant.
            "reset_robot_joints": EventTermCfg(
                mode="reset",
                func=mdp.reset_joints_by_offset,
                params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
            ),
        },
        curriculum_cfg={
            "command_vel": CurriculumTermCfg(
                func=mdp.commands_vel,
                params={
                    "command_name": "twist",
                    "velocity_stages": _command_vel_stages(),
                },
            ),
            **(extra_curriculum or {}),
        },
        metrics_cfg={
            # Logged to ``Episode_Metrics/<name>``; values mirror the reference's
            # ``Metrics/<name>_mean`` cross-(env, foot) writes from inside the
            # reward bodies (see ``.../velocity/mdp/rewards.py:205, 229,
            # 317, 352, 373``). GeneLab's MetricsManager averages over the
            # episode and reports at reset; the reference averages over the rollout via
            # rsl-rl. Per-step scalars match; the time horizons differ.
            "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
            "angular_momentum_mean": MetricsTermCfg(
                func=mdp.angular_momentum_mean,
                params={"sensor_name": "root_angmom"},
            ),
            "air_time_mean": MetricsTermCfg(
                func=mdp.air_time_mean,
                params={"sensor_name": "feet_ground_contact"},
            ),
            "slip_velocity_mean": MetricsTermCfg(
                func=mdp.slip_velocity_mean,
                params={
                    "sensor_name": "feet_ground_contact",
                    "asset_cfg": _feet_cfg(),
                },
            ),
            "landing_force_mean": MetricsTermCfg(
                func=mdp.landing_force_mean,
                params={"sensor_name": "feet_ground_contact"},
            ),
            "peak_height_mean": MetricsTermCfg(
                func=mdp.peak_height_mean,
                params={
                    "sensor_name": "feet_ground_contact",
                    "asset_cfg": _feet_cfg(),
                },
            ),
        },
    )

    if not play:
        # --- runtime DR (interval-mode push) ---
        cfg.events_cfg["push_robot"] = EventTermCfg(
            mode="interval",
            interval_range_s=(1.0, 3.0),
            func=mdp.push_by_setting_velocity,
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        )
        # --- sim2real startup DR (sampled once per env at training start) ---
        # Foot friction: only the foot links get perturbed; ``shared_random=True`` so both
        # feet of a given env get the same coefficient (modelling a single ground material).
        cfg.events_cfg["foot_friction"] = EventTermCfg(
            mode="startup",
            func=mdp.dr.geom_friction,
            params={
                "asset_cfg": _feet_cfg(),
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        )
        # Pelvis COM offset: the reference uses torso_link in its config but the comment in
        # tasks/velocity/config/g1/env_cfgs.py points at the same body.
        cfg.events_cfg["base_com"] = EventTermCfg(
            mode="startup",
            func=mdp.dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", link_names=(_TORSO_LINK,)),
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
            },
        )
        # Per-joint encoder bias: small constant offset between policy-perceived and
        # physical joint angles, applied symmetrically on obs (``joint_pos_rel``) and
        # action (``JointPositionAction.process_actions``) so the policy must be robust
        # to a silently shifted zero-point.
        cfg.events_cfg["encoder_bias"] = EventTermCfg(
            mode="startup",
            func=mdp.dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg(name="robot"),
                "bias_range": (-0.015, 0.015),
            },
        )

    if play:
        # Interactive teleop: don't auto-reset the lone robot every 20 s (time-out) or on a
        # stumble — let it keep walking under slider/keyboard control until the viewer closes.
        cfg.auto_reset = False

        # Frame the viewer on the single play robot. ``camera_lookat`` is also the trackball
        # pivot, so aiming it at the robot is what lets the mouse-wheel zoom close in on it
        # (the default pivot is the scene centroid — for the 80 m rough terrain that sits
        # tens of metres from the lone robot, which reads as "can't zoom in"). On flat ground
        # the robot spawns at the origin; on the heightfield it spawns at a fixed terrain
        # cell, so frame that.
        look_x, look_y = _single_play_spawn_xy(terrain)
        cfg.simulation.camera_lookat = (look_x, look_y, 0.6)
        cfg.simulation.camera_pos = (look_x + 3.0, look_y - 3.0, 1.7)

        # Auto-attach the in-viewport ImGui teleop bridge: three sliders (vx, vy, ωz) that
        # drive the single robot's ``twist`` command. Guarded on ``imgui_bundle`` (the
        # overlay's dependency, the 'imgui' extra) so a user without it still gets a working
        # play path — just no sliders.
        try:
            import imgui_bundle  # noqa: F401  # the ImGui overlay needs it at draw time

            from genelab.bridges.imgui import ImGuiTwistBridgeCfg

            cfg.simulation.viewer_imgui = True  # enable the overlay that hosts the sliders
            cfg.bridges_cfg["teleop"] = ImGuiTwistBridgeCfg(
                command_name="twist",
                vx_range=(-2.0, 2.0),
                vy_range=(-1.0, 1.0),
                wz_range=(-0.7, 0.7),
                # The training mix has only 10% standing envs with 3-8 s resample,
                # so the policy never sees ``command=[0, 0, 0]`` sustained for a
                # full episode. Start the slider in the forward-env training slice
                # (vx ≥ 0.3, vy = 0, ωz = 0) so iter-0 is unambiguously
                # in-distribution; drag the vx slider down to test standing.
                default_vx=0.3,
                default_vy=0.0,
                default_wz=0.0,
            )
        except ImportError:
            pass

    return cfg


def unitree_g1_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Flat-ground velocity-tracking env config for the Unitree G1."""
    return _velocity_env_cfg_base(play=play)
