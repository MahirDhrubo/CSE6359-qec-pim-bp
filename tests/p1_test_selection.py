# OWNER: P1
import pytest

from qdec.contracts import FULL_GRID, QUICK_GRID
from qdec.workload.selection import select_workloads


def test_full_profile_unchanged_and_reduced_profile_explicit():
    assert select_workloads(FULL_GRID) == FULL_GRID
    reduced = select_workloads(FULL_GRID, max_distance=9, shots=1000)
    assert len(reduced) == 15 and all(w.shots == 1000 for w in reduced)
    assert sum(w.code == "gross" for w in reduced) == 3
    assert FULL_GRID[0].shots == 100_000  # never mutate shared constants
    assert select_workloads(QUICK_GRID, shots=100_000) == QUICK_GRID


@pytest.mark.parametrize("kwargs", [{"shots": 0}, {"max_distance": 1}, {"code": "unknown"}])
def test_invalid_selection(kwargs):
    with pytest.raises(ValueError):
        select_workloads(FULL_GRID, **kwargs)
