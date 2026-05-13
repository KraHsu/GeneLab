"""Module-execution entry point.

Enables ``python -m genelab.cli ...`` (and therefore
``torchrun -m genelab.cli train TASK_ID ...``) to dispatch the same Typer app
exposed by the ``genelab`` console script.
"""

from genelab.cli import main

if __name__ == "__main__":
    main()
