"""``genelab export`` — CLI driver for TorchScript / ONNX policy export.

Resolves the task, builds the play env, loads the checkpoint via the backend's
``make_inference_setup``, extracts the bare actor ``nn.Module`` and the policy
obs-group schema, and delegates serialization to
:func:`genelab.rl.exporter.export_policy`.
"""

from pathlib import Path
from typing import Any, Literal

from genelab.cache import ensure_project_cache
from genelab.registry import TASKS
from genelab.rl.backends.base import PlayContext, ProfileArgs
from genelab.rl.exporter import ExportConfig, export_policy
from genelab.rl.runner import build_env, resolve_env_cfg


def export_task(
    task_id: str,
    checkpoint: Path,
    *,
    format: Literal["torchscript", "onnx"] = "torchscript",
    output: Path,
    opset: int = 17,
    num_envs: int = 1,
) -> Path:
    """Build the export artifact for ``task_id`` from ``checkpoint``.

    ``num_envs`` is the env-batch size used only to probe obs shapes; the exported
    model uses dynamic batch dimension via ONNX ``dynamic_axes``.
    """
    from genelab.rl.backends import select_backend

    ensure_project_cache()
    task = TASKS.get(task_id)
    task_cfg = getattr(task, "cfg", None)
    if task_cfg is None:
        raise SystemExit(f"task {task_id!r} has no .cfg attribute")
    agent_cfg = getattr(task_cfg, "agent", None)
    if agent_cfg is None:
        raise SystemExit(
            f"task {task_id!r} did not register an agent cfg; export requires a trainable task"
        )

    env_cfg = resolve_env_cfg(task_id, play=True)
    env_cfg.simulation.num_envs = int(num_envs)
    # Export only probes obs shapes — it never renders. Force ``vis=False`` so
    # tasks whose ``play_env`` was configured for viewer playback still export
    # on CI / remote servers without a display.
    env_cfg.simulation.vis = False
    env = build_env(env_cfg)

    backend = select_backend(agent_cfg)
    ctx = PlayContext(
        task_id=task_id,
        env=env,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        checkpoint=checkpoint,
        kind="trained",
        deterministic=True,
        max_steps=None,
        bridges=[],
        profile=ProfileArgs(),
    )

    try:
        setup = backend.make_inference_setup(ctx)
        if setup.actor_module is None:
            raise SystemExit(
                f"backend {backend.name!r} did not expose an actor module; export is not "
                "supported for this checkpoint type"
            )
        policy_group = _policy_group_name(agent_cfg)
        her_obs_group, goal_groups = _her_groups(agent_cfg)
        if her_obs_group is not None:
            policy_group = her_obs_group
        action_dim = int(env.action_manager.total_action_dim)
        cfg = ExportConfig(format=format, output=Path(output), opset=opset)
        return export_policy(
            task_id=task_id,
            env=env,
            checkpoint=checkpoint,
            actor=setup.actor_module,
            actor_input_dim=setup.actor_input_dim or 0,
            action_dim=action_dim,
            policy_group=policy_group,
            goal_groups=goal_groups,
            cfg=cfg,
        )
    finally:
        env.close()


def _policy_group_name(agent_cfg: Any) -> str:
    """Return the obs group fed to the policy network for ``agent_cfg``.

    Handles backend-specific cfg shapes:

    * rsl_rl: ``obs_groups["actor"]`` is a tuple of group names; we take the first.
    * skrl / sb3: ``obs_group`` is a single string field.
    """
    obs_group = getattr(agent_cfg, "obs_group", None)
    if isinstance(obs_group, str):
        return obs_group
    obs_groups = getattr(agent_cfg, "obs_groups", None)
    if isinstance(obs_groups, dict):
        actor_groups = obs_groups.get("actor")
        if actor_groups:
            return str(actor_groups[0])
    return "policy"


def _her_groups(agent_cfg: Any) -> tuple[str | None, list[str]]:
    """Return ``(policy_obs_group, goal_groups)`` for goal-conditioned SB3 tasks.

    SAC+HER exposes a ``Dict`` observation (``observation`` / ``achieved_goal`` /
    ``desired_goal``); the exporter concatenates these groups into the flat ``obs``
    input so the actor can rebuild the Dict. Returns ``(None, [])`` for non-HER
    agents (flat single-group export).
    """
    her = getattr(agent_cfg, "her", None)
    if her is not None and getattr(her, "enabled", False):
        return getattr(her, "obs_group", None), [
            her.achieved_goal_group,
            her.desired_goal_group,
        ]
    return None, []
