"""Module-execution entry point.

Enables ``python -m genelab.cli ...`` (and therefore
``torchrun -m genelab.cli train TASK_ID ...``) to dispatch the same Typer app
exposed by the ``genelab`` console script.
"""

from genelab.cli._bootstrap import pin_visible_device_for_rank

pin_visible_device_for_rank()

from genelab.cli import main  # noqa: E402  (must follow pin_visible_device_for_rank)

if __name__ == "__main__":
    main()
