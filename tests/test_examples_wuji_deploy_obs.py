"""Policy observation assembly for deploy (pure numpy, no simulator).

The deployed obs needs no forward kinematics: joint state comes from encoders,
cube/goal poses come from the observer (already in the tag frame), and the last
action is tracked. ``DeployObsBuilder`` reproduces the GeneLab training policy obs
(term order, per-term 3-step history, 6D goal-error encoding) so an exported ONNX
policy receives exactly what it was trained on.

Policy obs layout (matches ``genelab_wuji.reorient.env_cfg`` policy group):
    joint_pos_rel_history   20 * 3 = 60
    joint_vel_rel_history   20 * 3 = 60
    cube_pos_in_tag_history  3 * 3 =  9
    goal_rot_err_6d_history  6 * 3 = 18
    last_action_history     20 * 3 = 60
                                   = 207
"""

import numpy as np
import pytest

from genelab_wuji.deploy.obs import DeployObsBuilder, goal_rot_err_6d

_N_JOINTS = 20
_HIST = 3
_OBS_DIM = 207
_JP, _JV, _CUBE, _GOAL, _ACT = 60, 60, 9, 18, 60


def _default_qpos() -> np.ndarray:
    return np.linspace(0.1, 0.9, _N_JOINTS)


def test_obs_dim_is_207() -> None:
    builder = DeployObsBuilder(default_joint_pos=_default_qpos(), history_len=_HIST)
    builder.reset()
    obs = builder.compute(
        joint_pos=_default_qpos(),
        joint_vel=np.zeros(_N_JOINTS),
        cube_pos_tag=np.zeros(3),
        cube_quat_tag=np.array([1.0, 0.0, 0.0, 0.0]),
        goal_quat_tag=np.array([1.0, 0.0, 0.0, 0.0]),
        last_action=np.zeros(_N_JOINTS),
    )
    assert obs.shape == (_OBS_DIM,)


def test_first_frame_backfills_history() -> None:
    # After reset, the first compute should fill all 3 history slots with the same
    # frame (mirrors the training CircularBuffer backfill on reset).
    builder = DeployObsBuilder(default_joint_pos=_default_qpos(), history_len=_HIST)
    builder.reset()

    delta = 0.05 * np.ones(_N_JOINTS)
    obs = builder.compute(
        joint_pos=_default_qpos() + delta,
        joint_vel=np.zeros(_N_JOINTS),
        cube_pos_tag=np.zeros(3),
        cube_quat_tag=np.array([1.0, 0.0, 0.0, 0.0]),
        goal_quat_tag=np.array([1.0, 0.0, 0.0, 0.0]),
        last_action=np.zeros(_N_JOINTS),
    )

    # joint_pos_rel block = (joint_pos - default) repeated across 3 frames.
    jp_block = obs[:_JP].reshape(_HIST, _N_JOINTS)
    assert np.allclose(jp_block[0], delta)
    assert np.allclose(jp_block[1], delta)
    assert np.allclose(jp_block[2], delta)


def test_history_rolls_oldest_to_newest() -> None:
    builder = DeployObsBuilder(default_joint_pos=_default_qpos(), history_len=_HIST)
    builder.reset()

    act_start = _JP + _JV + _CUBE + _GOAL  # last_action block offset

    def step(action_val: float) -> np.ndarray:
        obs = builder.compute(
            joint_pos=_default_qpos(),
            joint_vel=np.zeros(_N_JOINTS),
            cube_pos_tag=np.zeros(3),
            cube_quat_tag=np.array([1.0, 0.0, 0.0, 0.0]),
            goal_quat_tag=np.array([1.0, 0.0, 0.0, 0.0]),
            last_action=action_val * np.ones(_N_JOINTS),
        )
        return obs[act_start : act_start + _ACT].reshape(_HIST, _N_JOINTS)[:, 0]

    step(1.0)  # backfill -> [1, 1, 1]
    step(2.0)  # roll     -> [1, 1, 2]
    frames = step(3.0)  # roll -> [1, 2, 3]
    assert np.allclose(frames, [1.0, 2.0, 3.0])  # oldest -> newest


def test_goal_rot_err_6d_matches_genelab_training_math() -> None:
    # Pin the 6D encoding against the *actual* GeneLab training code path
    # (genelab.utils.math + reorient.mdp._math), the policy was trained on this.
    torch = pytest.importorskip("torch")
    from genelab.utils.math import matrix_from_quat, quat_conjugate, quat_mul
    from genelab_wuji.reorient.mdp._math import matrix_to_rotation_6d

    rng = np.random.default_rng(3)
    for _ in range(8):
        cube = rng.standard_normal(4)
        cube /= np.linalg.norm(cube)
        goal = rng.standard_normal(4)
        goal /= np.linalg.norm(goal)

        ours = goal_rot_err_6d(cube, goal)

        c = torch.tensor(cube, dtype=torch.float).unsqueeze(0)
        g = torch.tensor(goal, dtype=torch.float).unsqueeze(0)
        err = quat_mul(c, quat_conjugate(g))
        ref = matrix_to_rotation_6d(matrix_from_quat(err)).squeeze(0).numpy()

        assert np.allclose(ours, ref, atol=1e-5), f"{ours} != {ref}"
