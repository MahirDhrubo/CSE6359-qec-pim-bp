# OWNER: P3 — placeholder; replace with real tests.
import numpy as np
import pytest


@pytest.mark.xfail(reason="not implemented", raises=NotImplementedError)
def test_p3_smoke():
    from qdec.queue.sim import simulate
    from qdec.results import plots  # noqa: F401

    simulate(np.ones(10), 1.0)
