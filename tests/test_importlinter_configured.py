"""Smoke test: the importlinter contracts stay configured (ADR-0009 R9.6).

The layering gate is only as strong as its config. If the ``[tool.importlinter]``
section were accidentally deleted (or a contract dropped), ``lint-imports`` would
report "0 contracts, 0 broken" and pass *vacuously* — silently disabling the
required CI check flipped on in R7.3d. This test asserts the expected contracts are
present so that regression fails the suite instead.
"""

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# The full contract set as of R7.3d (ADR-0009). Adding a contract is fine; removing
# or renaming one of these must be a deliberate edit that updates this set too.
_EXPECTED_CONTRACTS = frozenset(
    {
        "Domain modules are below cli / rl",
        "Domain (except asset_zoo) does not import utils.download",
        "rl.backends does not import rl.runner",
        "rl is below cli",
        "Infrastructure modules do not import up",
        "RL backends do not import each other",
    }
)


def _importlinter_config() -> dict:
    data = tomllib.loads(_PYPROJECT.read_text())
    return data["tool"]["importlinter"]


def test_importlinter_contracts_present() -> None:
    config = _importlinter_config()
    names = {contract["name"] for contract in config["contracts"]}
    missing = _EXPECTED_CONTRACTS - names
    assert not missing, f"importlinter contracts missing from pyproject.toml: {sorted(missing)}"


def test_importlinter_roots_and_type_checking_filter() -> None:
    config = _importlinter_config()
    assert config["root_packages"] == ["genelab"]
    # Domain modules import ManagerBasedRlEnv under ``if TYPE_CHECKING`` for type hints;
    # this filter keeps those from registering as runtime cross-layer violations.
    assert config["exclude_type_checking_imports"] is True
