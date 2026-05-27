"""Smoke test: ``examples/genelab_showcase`` registers every play-only task.

No env build, no Genesis import. The test loads the showcase extension by module
path (``--import genelab_showcase.tasks``) and checks the registry contents — keeping
the test fast and runnable on machines without CUDA.
"""

from genelab.cli import main
from genelab.registry import TASKS

_EXPECTED_TASK_IDS: tuple[str, ...] = (
    "GeneLab-Sensors-Showcase-v0",
    "GeneLab-RayCast-Showcase-v0",
    "GeneLab-Contact-Showcase-v0",
    "GeneLab-Terrain-Showcase-v0",
    "GeneLab-Curriculum-Showcase-v0",
    "GeneLab-Actuator-Showcase-v0",
    "GeneLab-MlpResidual-Actuator-Showcase-v0",
    "GeneLab-Recording-Showcase-v0",
)


def test_showcase_extension_registers_all_tasks() -> None:
    main(["--no-entry-points", "--import", "genelab_showcase.tasks", "list", "tasks"])
    registered = set(TASKS.names())
    missing = [t for t in _EXPECTED_TASK_IDS if t not in registered]
    assert not missing, f"showcase tasks not registered: {missing}"


def test_showcase_extension_is_idempotent() -> None:
    # Loading the extension twice must not raise a duplicate-registration error.
    main(["--no-entry-points", "--import", "genelab_showcase.tasks", "list", "tasks"])
    main(["--no-entry-points", "--import", "genelab_showcase.tasks", "list", "tasks"])
    registered = set(TASKS.names())
    for task_id in _EXPECTED_TASK_IDS:
        assert task_id in registered, f"missing after re-import: {task_id}"
