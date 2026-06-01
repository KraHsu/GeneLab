"""A "recall the target" cart-pole — a task that genuinely *requires* memory.

Unlike the POMDP balancing variants (which a memoryless policy still solves, because
feedback stabilisation is robust to unobserved state), this task hides information that
**cannot be recovered by feedback**: a per-episode cart target shown only for the first
``cue_steps`` (a brief flash) and then removed from the observation. The cue is too short to
drive the cart to the target while it is visible, so the cart is still near the centre when
the target vanishes — the policy must have *remembered* it to move there afterwards. A
memoryless MLP, once the cue is gone, has no idea where the target is and falls back to a
fixed point (the mean target, ~centre); a recurrent policy stores the target during the
flash and drives the cart to it. The cart-to-target gap between the two is the demonstration.

The cue is deliberately brief and there are no disturbances, so the task is *learnable*
(constant pushes made earlier versions hopeless) while still being unsolvable without
memory. Velocities stay observed — the only thing needing memory is the target, isolating
the recurrence benefit cleanly.
"""

from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg

from genelab_inverted_pendulum import mdp as ip_mdp
from genelab_inverted_pendulum.single.env_cfg import inverted_pendulum_env_cfg

_CUE_STEPS = 5  # brief flash: long enough to read the target, too short to drive there


def inverted_pendulum_memory_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Cart-pole that must drive the cart to a target flashed only at the episode start."""
    cfg = inverted_pendulum_env_cfg(play=play)
    # Short episodes (~100 control steps) keep the cue→reach memory dependency inside one
    # truncated-BPTT window, so the recurrent gradient can actually learn it (see rnn_cfg).
    cfg.episode_length_s = 1.0

    # Add the (masked) target cue to both observation groups.
    cfg.observations_cfg["policy"].terms["target_cue"] = ObservationTermCfg(
        func=ip_mdp.cart_target_cue, params={"cue_steps": _CUE_STEPS}
    )
    cfg.observations_cfg["critic"].terms["target_cue"] = ObservationTermCfg(
        func=ip_mdp.cart_target_cue, params={"cue_steps": _CUE_STEPS}
    )

    # Track the hidden target instead of holding the cart at the origin; weight it heavily so
    # the policy clearly prioritises reaching the remembered target over sitting at centre.
    cfg.rewards_cfg.pop("cart_position", None)
    cfg.rewards_cfg["cart_target"] = RewardTermCfg(func=ip_mdp.cart_target_tracking_l2, weight=-4.0)

    # Sample a fresh hidden target each reset. No pushes — the brief cue already prevents the
    # cart from reaching (and parking at) the target while it is still visible.
    cfg.events_cfg["reset_cart_target"] = EventTermCfg(
        mode="reset",
        func=ip_mdp.reset_cart_target,
        params={"target_range": (-0.8, 0.8)},  # reachable within a 1 s episode
    )
    cfg.events_cfg.pop("push_cart", None)
    return cfg
