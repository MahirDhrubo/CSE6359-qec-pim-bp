# OWNER: P1 — placeholder; replace with real tests.
import pytest


@pytest.mark.xfail(reason="not implemented", raises=NotImplementedError)
def test_p1_smoke():
    from qdec.contracts import QUICK_GRID
    from qdec.workload.circuits import make_circuit

    make_circuit(QUICK_GRID[0])
