"""M4 end-to-end smoke: mount IMU + FrameTransformer + RayCast(RingPattern) on a real env.

Builds the single inverted-pendulum task in play mode (``num_envs=1``, ``vis=False``,
``device="cpu"``), augments ``scene.sensors`` with the three M4 sensors, runs a handful of
control steps through ``ManagerBasedRlEnv``, and asserts that each sensor's ``.data`` is
populated with the expected shape. Skips on hosts without ``libEGL`` (same guard as
``test_interactive_scene.py``).
"""

from typing import Any

import torch

from genelab.sensor import (
    BodyVelocitySensorCfg,
    FrameTransformerSensorCfg,
    HemispherePattern,
    IMUSensorCfg,
    RayCastSensorCfg,
    RingPattern,
    TargetFrameCfg,
)

CART_LINK = "cart"
POLE_LINK = "pole"


def test_m4_sensors_integrate_with_manager_based_rl_env(genesis_runtime: Any) -> None:
    del genesis_runtime  # fixture only guards Genesis availability; module not needed
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab_inverted_pendulum.single.env_cfg import inverted_pendulum_env_cfg

    cfg = inverted_pendulum_env_cfg(play=True)
    cfg.simulation.vis = False
    cfg.simulation.gpu = False
    cfg.device = "cpu"

    # Keep the existing pole_ang_vel sensor (the env's observation manager uses it) and add
    # the three M4 sensors. RingPattern probes the cart's local frame; HemispherePattern on
    # the IMU isn't applicable here, so we stick to a ring + FrameTransformer + IMU trio.
    cfg.scene.sensors = (
        BodyVelocitySensorCfg(name="pole_ang_vel", link_name=POLE_LINK, measure="ang_vel"),
        IMUSensorCfg(name="cart_imu", link_name=CART_LINK, gravity_bias=True),
        FrameTransformerSensorCfg(
            name="pole_in_cart",
            source_link_name=CART_LINK,
            target_frames=(TargetFrameCfg(link_name=POLE_LINK, name="pole_tip"),),
        ),
        RayCastSensorCfg(
            name="cart_lidar",
            link_name=CART_LINK,
            pattern=RingPattern(
                num_horizontal=4,
                num_vertical=1,
                horizontal_fov_deg=(-180.0, 180.0),
                vertical_fov_deg=(-30.0, -30.0),
            ),
            max_distance=5.0,
        ),
    )

    env = ManagerBasedRlEnv(cfg)
    try:
        env.reset()
        zero_actions = torch.zeros(
            env.num_envs, env.action_manager.total_action_dim, device=env.device
        )
        for _ in range(5):
            env.step(zero_actions)

        imu = env.sensors["cart_imu"].data
        assert imu.orientation.shape == (1, 4)
        assert imu.projected_gravity_b.shape == (1, 3)
        assert imu.lin_acc_b.shape == (1, 3)
        assert imu.ang_acc_b.shape == (1, 3)

        ft = env.sensors["pole_in_cart"].data
        assert ft.target_pos_w.shape == (1, 1, 3)
        assert ft.target_quat_w.shape == (1, 1, 4)
        assert ft.target_pos_source.shape == (1, 1, 3)
        assert ft.target_quat_source.shape == (1, 1, 4)
        assert env.sensors["pole_in_cart"].target_names == ("pole_tip",)

        ray = env.sensors["cart_lidar"].data
        assert ray.distances.shape == (1, 4)
        assert ray.hit_pos_w.shape == (1, 4, 3)
        assert ray.normals_w.shape == (1, 4, 3)
    finally:
        env.close()


def test_m4_hemisphere_pattern_integrates_with_manager_based_rl_env(
    genesis_runtime: Any,
) -> None:
    """Companion to the RingPattern check: build a env with HemispherePattern instead."""
    del genesis_runtime
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab_inverted_pendulum.single.env_cfg import inverted_pendulum_env_cfg

    cfg = inverted_pendulum_env_cfg(play=True)
    cfg.simulation.vis = False
    cfg.simulation.gpu = False
    cfg.device = "cpu"
    cfg.scene.sensors = (
        BodyVelocitySensorCfg(name="pole_ang_vel", link_name=POLE_LINK, measure="ang_vel"),
        RayCastSensorCfg(
            name="cart_dome",
            link_name=CART_LINK,
            pattern=HemispherePattern(
                num_rays_target=8,
                pole_axis=(0.0, 0.0, -1.0),
                polar_fov_deg=60.0,
            ),
            max_distance=5.0,
        ),
    )

    env = ManagerBasedRlEnv(cfg)
    try:
        env.reset()
        zero_actions = torch.zeros(
            env.num_envs, env.action_manager.total_action_dim, device=env.device
        )
        for _ in range(3):
            env.step(zero_actions)
        dome = env.sensors["cart_dome"].data
        assert dome.distances.shape == (1, 8)
        # Downward hemisphere on a cart at ~0.12 m above the ground: all rays should hit
        # the flat plane within max_distance.
        assert (dome.distances < 5.0).all()
    finally:
        env.close()
