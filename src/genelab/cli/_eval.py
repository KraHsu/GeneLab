"""``genelab eval`` — deterministic rollout evaluation that writes ``eval.json``.

The implementation moved to :func:`genelab.rl.eval_task.eval_task` (ADR-0009 /
ROADMAP §9 R7.3): it is backend-agnostic eval orchestration whose dependencies all
live in the ``rl`` / config bands, and both the ``genelab eval`` command *and* the
in-training ``EvalCallback`` call it — so it must sit in ``rl``, not above it in
``cli``. This module re-exports it as the CLI-facing name.
"""

from genelab.rl.eval_task import eval_task as eval_task
