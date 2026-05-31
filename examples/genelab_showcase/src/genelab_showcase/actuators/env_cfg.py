"""Actuator showcase envs: Franka arms on force-channel actuators.

* :func:`actuator_showcase_env_cfg` — one Franka whose seven arm joints use an
  :class:`IdealPDActuator` (force-channel PD) instead of the default implicit PD. The
  actuator zeros Genesis's internal PD gains at bind time and drives the joints by writing
  torque through ``control_dofs_force`` every sub-step.
* :func:`mlp_residual_actuator_showcase_env_cfg` — *two* Frankas side by side sharing one
  scripted target: ``robot`` runs an :class:`MlpResidualActuator` (DC-motor base + learned
  residual) and ``robot_ideal`` runs the plain :class:`IdealPDActuator`. Driving both through
  identical action pipelines lets the runner overlay "with residual" vs "without residual".

Both run at a fine sim dt (200 Hz) with decimation 4 (50 Hz control): the Python force-channel
PD is under-damped and buzzes at the coarser 0.01 dt, so the finer step plus a lower arm damping
keep it genuinely tracking. The hand keeps :class:`ImplicitPDActuator` (a stiff position task
Genesis's internal solver handles well).
"""

from genelab import mdp
from genelab.actuator import (
    ActuatorBaseCfg,
    IdealPDActuatorCfg,
    ImplicitPDActuatorCfg,
    MlpResidualActuatorCfg,
)
from genelab.asset_zoo import FrankaPandaCfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.entity import ArticulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import EventTermCfg, TerminationTermCfg
from genelab.mdp.actions.joint_position import JointPositionActionCfg

from genelab_showcase.actuators.residual_net import residual_net_path

# Stabilised force-channel timing, shared by both showcases (see module docstring).
_SIM_DT = 0.005
_DECIMATION = 4
_ARM_DAMPING = 40.0  # lower than the implicit-PD default (80) so the Python PD does not buzz


def _ideal_arm() -> IdealPDActuatorCfg:
    return IdealPDActuatorCfg(
        target_names_expr=(r"^joint[1-7]$",),
        stiffness=400.0,
        damping=_ARM_DAMPING,
        effort_limit=87.0,
        velocity_limit=2.175,
        action_scale=0.5,
    )


def _residual_arm() -> MlpResidualActuatorCfg:
    return MlpResidualActuatorCfg(
        target_names_expr=(r"^joint[1-7]$",),
        stiffness=400.0,
        damping=_ARM_DAMPING,
        effort_limit=87.0,
        velocity_limit=2.175,
        saturation_effort=87.0,
        action_scale=0.5,
        network_file=residual_net_path(),
        residual_scale=1.0,
    )


def _hand() -> ImplicitPDActuatorCfg:
    return ImplicitPDActuatorCfg(
        target_names_expr=(r"finger_joint.*",),
        stiffness=1.0e4,
        damping=200.0,
        effort_limit=20.0,
        velocity_limit=0.2,
        action_scale=0.04,
    )


def _franka(arm: ActuatorBaseCfg, init_pos: tuple[float, float, float]) -> ArticulationCfg:
    robot = FrankaPandaCfg()
    robot.init_pos = init_pos
    robot.actuators = {"panda_arm": arm, "panda_hand": _hand()}
    return robot


def _arm_action(asset_name: str) -> JointPositionActionCfg:
    return JointPositionActionCfg(
        asset_name=asset_name, joint_names=(r"^joint[1-7]$",), use_default_offset=True
    )


def _hand_action(asset_name: str) -> JointPositionActionCfg:
    return JointPositionActionCfg(
        asset_name=asset_name, joint_names=(r"finger_joint.*",), use_default_offset=True
    )


def _sim(num_envs: int) -> SimulationCfg:
    return SimulationCfg(
        num_envs=num_envs,
        dt=_SIM_DT,
        substeps=1,
        steps=400,
        vis=False,
        gpu=True,
    )


def actuator_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """One Franka with IdealPD on the arm (force-channel control)."""

    return ManagerBasedRlEnvCfg(
        simulation=_sim(num_envs=1),
        scene=InteractiveSceneCfg(env_spacing=(2.0, 2.0)),
        decimation=_DECIMATION,
        # Long episode so the scripted sweep runs continuously under the viewer (no auto-reset).
        episode_length_s=1000.0,
        device="cuda",
        robot=_franka(_ideal_arm(), init_pos=(0.0, 0.0, 0.0)),
        actions_cfg={
            "panda_arm": _arm_action("robot"),
            "panda_hand": _hand_action("robot"),
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


def mlp_residual_actuator_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """Two Frankas sharing one scripted target: ``robot`` = MlpResidual, ``robot_ideal`` = IdealPD.

    Both arms run identical action pipelines (same gains, scale and offset), so feeding the
    same per-step action drives the same commanded target — the only difference is the learned
    residual on ``robot``. The runner overlays the two joint-1 responses to show the residual's
    effect. ``robot_ideal`` is offset in y so both are visible.
    """

    return ManagerBasedRlEnvCfg(
        simulation=_sim(num_envs=1),
        scene=InteractiveSceneCfg(env_spacing=(2.0, 2.0)),
        decimation=_DECIMATION,
        episode_length_s=1000.0,
        device="cuda",
        robots={
            "robot": _franka(_residual_arm(), init_pos=(0.0, 0.0, 0.0)),
            "robot_ideal": _franka(_ideal_arm(), init_pos=(0.0, 1.2, 0.0)),
        },
        actions_cfg={
            "resid_arm": _arm_action("robot"),
            "resid_hand": _hand_action("robot"),
            "ideal_arm": _arm_action("robot_ideal"),
            "ideal_hand": _hand_action("robot_ideal"),
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
