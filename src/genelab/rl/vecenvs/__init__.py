"""VecEnv adapters (env -> RL-library) for each backend.

One module per library, named to match its sibling trainer under
``rl/backends/`` — ``rl/vecenvs/sb3.py`` adapts the env for the trainer in
``rl/backends/sb3.py`` (ADR-0007 / ROADMAP §9 Phase R6). The adapter classes
(``RslRlVecEnvWrapper``, ``GenelabSb3VecEnv``, ``GenelabSkrlWrapper``) keep their
names; only their import paths moved here from ``rl/<lib>_wrapper.py`` (now thin
deprecation shims).
"""

__all__: list[str] = []
