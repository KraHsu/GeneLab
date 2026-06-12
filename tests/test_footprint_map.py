"""Unit tests for the spatial footprint memory map (``genelab.terrains.footprint_map``).

Genesis-free. The map is the world-keyed terrain memory (ADR-0001 Q3, paper §6.5): a
footprint left at a world position is read back by *any* foot that later visits it.
"""

import pytest

torch = pytest.importorskip("torch")

from genelab.terrains.footprint_map import FootprintMap  # noqa: E402


def test_map_remembers_residual_by_world_position() -> None:
    m = FootprintMap(num_envs=1, resolution=0.1, size=(2.0, 2.0))
    here = torch.tensor([[[0.3, 0.4]]])  # (num_envs, num_points, 2)
    m.accumulate(here, torch.tensor([[0.05]]))
    assert float(m.read(here)) == pytest.approx(0.05)
    elsewhere = torch.tensor([[[-0.5, -0.5]]])
    assert float(m.read(elsewhere)) == pytest.approx(0.0)


def test_a_footprint_couples_feet_landing_in_the_same_cell() -> None:
    m = FootprintMap(num_envs=1, resolution=0.1, size=(2.0, 2.0))
    m.accumulate(torch.tensor([[[0.31, 0.41]]]), torch.tensor([[0.05]]))  # front foot stamps
    # rear foot lands 3 cm away — same 10 cm cell — and feels the footprint
    assert float(m.read(torch.tensor([[[0.34, 0.43]]]))) == pytest.approx(0.05)
    # one cell over: pristine ground
    assert float(m.read(torch.tensor([[[0.45, 0.41]]]))) == pytest.approx(0.0)


def test_footprints_recover_toward_zero_over_time() -> None:
    m = FootprintMap(num_envs=1, resolution=0.1, size=(2.0, 2.0))
    pos = torch.tensor([[[0.0, 0.0]]])
    m.accumulate(pos, torch.tensor([[0.1]]))
    m.recover(dt=0.1, recovery_time=1.0)  # relax by dt/tau = 10%
    assert float(m.read(pos)) == pytest.approx(0.09)


def test_reset_clears_selected_envs() -> None:
    m = FootprintMap(num_envs=2, resolution=0.1, size=(2.0, 2.0))
    pos = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]])  # (2, 1, 2)
    m.accumulate(pos, torch.tensor([[0.1], [0.2]]))
    m.reset(torch.tensor([0]))
    r = m.read(pos)
    assert float(r[0]) == 0.0
    assert float(r[1]) == pytest.approx(0.2)


def test_out_of_bounds_positions_clamp_to_the_edge() -> None:
    m = FootprintMap(num_envs=1, resolution=0.1, size=(2.0, 2.0))
    far = torch.tensor([[[100.0, 100.0]]])  # well outside the grid
    m.accumulate(far, torch.tensor([[0.1]]))  # clamps to the corner cell, no crash
    assert float(m.read(far)) == pytest.approx(0.1)
