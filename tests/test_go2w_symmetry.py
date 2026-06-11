"""Mirror-symmetry augmentation for the Go2-W velocity task.

The Go2-W and its velocity task are mirror-symmetric about the x-z plane: flipping the
world left-right maps a +vy crab-walk onto a −vy crab-walk and a +wz turn onto a −wz turn.
``mirror_go2w_obs_actions`` exploits that to double every PPO mini-batch with mirrored
samples (rsl_rl ``Symmetry`` extension), which structurally prevents the one-sided gait
collapse observed across three stage-2 fine-tunes (each ended with one lateral direction
falling 64/64 while the other tracked perfectly).

Math being verified: under the y-flip, linear quantities negate their y component, angular
quantities negate x and z; left/right joints swap within each 4-group (FL,FR,RL,RR) and the
abduction (hip, x-axis) angles negate while thigh / calf / wheel (y-axis) keep sign.
"""

import pytest

torch = pytest.importorskip("torch")

from genelab_unitree.go2w.symmetry import (  # noqa: E402
    POLICY_FRAME_DIM,
    mirror_go2w_obs_actions,
)


def _obs_dict(num_envs: int = 4) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "policy": torch.randn(num_envs, POLICY_FRAME_DIM * 5),
        "critic": torch.randn(num_envs, (POLICY_FRAME_DIM + 3) * 5),
    }


def test_mirror_doubles_batch_and_is_involutive() -> None:
    obs = _obs_dict()
    actions = torch.randn(4, 16)
    obs_aug, act_aug = mirror_go2w_obs_actions(env=None, obs=obs, actions=actions)
    # [orig; mirrored] layout, double rows.
    assert obs_aug["policy"].shape == (8, POLICY_FRAME_DIM * 5)
    assert act_aug.shape == (8, 16)
    assert torch.equal(obs_aug["policy"][:4], obs["policy"])
    assert torch.equal(act_aug[:4], actions)
    # Mirroring the mirrored half recovers the original (involution).
    obs_2, act_2 = mirror_go2w_obs_actions(
        env=None,
        obs={k: v[4:] for k, v in obs_aug.items()},
        actions=act_aug[4:],
    )
    assert torch.allclose(obs_2["policy"][4:], obs["policy"], atol=1e-6)
    assert torch.allclose(act_2[4:], actions, atol=1e-6)


def test_mirror_flips_lateral_command_and_swaps_hips() -> None:
    obs = _obs_dict(num_envs=1)
    pol = obs["policy"]
    obs_aug, _ = mirror_go2w_obs_actions(env=None, obs=obs, actions=None)
    m = obs_aug["policy"][1]  # mirrored row
    # Frame 0 layout: ang_vel(0:3) gravity(3:6) joint_pos(6:22) joint_vel(22:38)
    # actions(38:54) commands(54:57).
    # Angular velocity: wx, wz negate; wy keeps sign.
    assert m[0] == -pol[0, 0] and m[1] == pol[0, 1] and m[2] == -pol[0, 2]
    # Gravity: gy negates.
    assert m[3] == pol[0, 3] and m[4] == -pol[0, 4] and m[5] == pol[0, 5]
    # joint_pos hips (6:10) = [FL,FR,RL,RR]: swapped pairs with negation.
    assert m[6] == -pol[0, 7] and m[7] == -pol[0, 6]
    assert m[8] == -pol[0, 9] and m[9] == -pol[0, 8]
    # joint_pos thighs (10:14): swapped, no negation.
    assert m[10] == pol[0, 11] and m[11] == pol[0, 10]
    # joint_pos wheels (18:22): swapped, no negation.
    assert m[18] == pol[0, 19] and m[19] == pol[0, 18]
    # Commands (54:57) = [vx, vy, wz]: vy and wz negate.
    assert m[54] == pol[0, 54] and m[55] == -pol[0, 55] and m[56] == -pol[0, 56]
    # Frame 1 repeats the same per-frame map at offset 57 (frame-major stack).
    assert m[57] == -pol[0, 57]


def test_mirror_critic_lin_vel_flips_y() -> None:
    obs = _obs_dict(num_envs=1)
    cr = obs["critic"]
    obs_aug, _ = mirror_go2w_obs_actions(env=None, obs=obs, actions=None)
    m = obs_aug["critic"][1]
    # Critic frame = policy frame + base_lin_vel(57:60): vy negates.
    assert m[57] == cr[0, 57] and m[58] == -cr[0, 58] and m[59] == cr[0, 59]


def test_mirror_action_swaps_legs_and_wheels() -> None:
    actions = torch.arange(16, dtype=torch.float32).unsqueeze(0)
    _, act_aug = mirror_go2w_obs_actions(env=None, obs=None, actions=actions)
    m = act_aug[1]
    # Action layout: leg position targets hip(0:4) thigh(4:8) calf(8:12), wheels(12:16).
    assert m[0] == -actions[0, 1] and m[1] == -actions[0, 0]  # hips swap + negate
    assert m[4] == actions[0, 5] and m[5] == actions[0, 4]  # thighs swap
    assert m[8] == actions[0, 9] and m[9] == actions[0, 8]  # calves swap
    assert m[12] == actions[0, 13] and m[13] == actions[0, 12]  # wheels swap
    assert m[14] == actions[0, 15] and m[15] == actions[0, 14]
