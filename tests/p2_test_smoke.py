# OWNER: P2 — placeholder; replace with real tests.
import pytest


@pytest.mark.xfail(reason="not implemented", raises=NotImplementedError)
def test_p2_smoke():
    from qdec.cache.cam import ClusterCache
    from qdec.costmodel import service  # noqa: F401

    ClusterCache()
