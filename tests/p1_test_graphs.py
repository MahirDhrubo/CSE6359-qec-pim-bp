# OWNER: P1
import numpy as np
import pytest
import scipy.sparse as sp
import stim

from qdec.contracts import QUICK_GRID
from qdec.workload.circuits import make_circuit
from qdec.workload.graphs import _merged_columns, dem_to_graph


@pytest.fixture(scope="module", params=QUICK_GRID, ids=lambda w: w.tag)
def built(request):
    c = make_circuit(request.param)
    return c, dem_to_graph(c)


def test_shapes_and_dtypes(built):
    c, g = built
    H, L, priors = g["H"], g["L"], g["priors"]
    n = len(priors)
    assert H.format == "csr" and L.format == "csr"
    assert H.shape == (c.num_detectors, n) and L.shape == (c.num_observables, n)
    assert H.dtype == np.uint8 and L.dtype == np.uint8
    assert set(np.unique(H.data)) <= {1} and set(np.unique(L.data)) <= {1}
    assert priors.dtype == np.float64 and ((priors > 0) & (priors < 0.5)).all()


def test_matches_dem_error_sampler(built):
    # Error vectors sampled from the DEM must reproduce its detectors/observables through H, L
    # (after XOR-folding duplicate DEM errors onto their merged column).
    c, g = built
    dem = c.detector_error_model(decompose_errors=False)
    dets, obs, errs = dem.compile_sampler(seed=7).sample(2_000, return_errors=True)
    _, col_of = _merged_columns(dem)
    fold = sp.csr_matrix(
        (np.ones(len(col_of), np.uint8), (col_of, np.arange(len(col_of)))),
        shape=(len(g["priors"]), len(col_of)),
    )
    e = (fold @ errs.astype(np.uint8).T) % 2
    np.testing.assert_array_equal((g["H"] @ e % 2).T.astype(bool), dets)
    np.testing.assert_array_equal((g["L"] @ e % 2).T.astype(bool), obs)


def test_columns_are_distinct(built):
    _, g = built
    cols = np.vstack([g["H"].toarray(), g["L"].toarray()]).T
    assert len(np.unique(cols, axis=0)) == cols.shape[0]


def test_hyperedges_kept(built):
    # decompose_errors=False: Y-type faults touch > 2 detectors (matching graph would split them).
    _, g = built
    assert g["H"].getnnz(axis=0).max() > 2


def test_duplicate_errors_merged_as_xor():
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0 L0\nerror(0.3) D0 D1\ndetector D1")
    g, col_of = _merged_columns(dem)
    assert g["H"].shape == (2, 2) and col_of.tolist() == [0, 0, 1]
    np.testing.assert_allclose(g["priors"], [0.1 + 0.2 - 2 * 0.1 * 0.2, 0.3])
    assert g["L"].toarray().tolist() == [[1, 0]]
