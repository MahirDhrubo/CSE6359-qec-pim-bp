# OWNER: P2
"""Day 1: cluster splitter, canonical keys and isolation check, on P2's own Stim shots."""
import numpy as np
import pytest
import scipy.sparse as sp

from qdec.cache.devshots import sample_dev_shots
from qdec.cache.keys import Boundary, canonical_key, is_isolated
from qdec.cache.splitter import Cluster, detector_adjacency, find_clusters
from qdec.contracts import SEED_TEST, SEED_TRAIN


@pytest.fixture(scope="module")
def shots():
    """d=5, p=1e-3 rotated surface code, TRAIN and TEST seeds (not P1's files)."""
    return {
        "train": sample_dev_shots(5, 1e-3, 2000, SEED_TRAIN),
        "test": sample_dev_shots(5, 1e-3, 2000, SEED_TEST),
    }


@pytest.fixture(scope="module")
def graph(shots):
    s = shots["train"]
    return detector_adjacency(s["H"]), Boundary.from_coords(s["det_coords"])


def _coord_index(det_coords):
    return {tuple(c): i for i, c in enumerate(det_coords.tolist())}


def _cluster_at(det_coords, coords):
    idx = _coord_index(det_coords)
    ids = sorted(idx[tuple(map(float, c))] for c in coords)
    return Cluster(det_ids=tuple(ids), coords=det_coords[ids])


# ---------------------------------------------------------------- fixture sanity
def test_seeds_give_different_shots(shots):
    assert not np.array_equal(shots["train"]["dets"], shots["test"]["dets"])


def test_dev_shots_schema(shots):
    s = shots["train"]
    S, D = s["dets"].shape
    assert s["dets"].dtype == bool and s["obs"].shape[0] == S
    assert s["det_coords"].shape == (D, 3) and s["H"].shape[0] == D


# ---------------------------------------------------------------- splitter
def test_adjacency_symmetric_no_self_loops(graph):
    A, _ = graph
    assert (A != A.T).nnz == 0
    assert not A.diagonal().any()


def test_find_clusters_toy_graph():
    # path 0-1-2, isolated 3, path 4-5
    H = sp.csr_matrix(np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
    ], dtype=np.uint8))
    A = detector_adjacency(H)
    coords = np.arange(6, dtype=float)[:, None]
    ev = np.array([1, 0, 1, 1, 1, 1], dtype=bool)   # 0 and 2 fired but 1 did not -> separate
    got = sorted(c.det_ids for c in find_clusters(ev, coords, A))
    assert got == [(0,), (2,), (3,), (4, 5)]
    assert find_clusters(np.zeros(6, bool), coords, A) == []


def test_clusters_partition_fired_detectors(shots, graph):
    A, _ = graph
    s = shots["test"]
    for row in s["dets"][:300]:
        cl = find_clusters(row, s["det_coords"], A)
        ids = [d for c in cl for d in c.det_ids]
        assert sorted(ids) == np.flatnonzero(row).tolist()
        for c in cl:
            assert list(c.det_ids) == sorted(c.det_ids)
            np.testing.assert_array_equal(c.coords, s["det_coords"][list(c.det_ids)])


def test_cluster_counts_sane_d5(shots, graph):
    A, _ = graph
    s = shots["test"]
    counts = [len(find_clusters(r, s["det_coords"], A)) for r in s["dets"]]
    sizes = [len(c.det_ids) for r in s["dets"] for c in find_clusters(r, s["det_coords"], A)]
    # at p=1e-3, d=5 most shots have a few small clusters
    assert 0.5 < np.mean(counts) < 5
    assert np.median(sizes) <= 3


# ---------------------------------------------------------------- keys
def test_boundary_geometry(graph):
    _, bd = graph
    np.testing.assert_array_equal(bd.step, [2, 2, 1])      # Stim rotated surface code spacing
    np.testing.assert_array_equal(bd.grid(bd.hi), [5, 5, 5])


def test_translated_pattern_same_key(shots, graph):
    _, bd = graph
    dc = shots["train"]["det_coords"]
    pattern = [(4, 4, 2), (4, 4, 3)]                         # a time-like pair in the bulk
    shifted = [(x + 2, y + 2, t) for x, y, t in pattern]     # (1,1) steps keeps the sublattice
    a, b = _cluster_at(dc, pattern), _cluster_at(dc, shifted)
    assert a.det_ids != b.det_ids
    assert canonical_key(a, bd) == canonical_key(b, bd)


def test_translated_pattern_same_key_d9():
    s = sample_dev_shots(9, 1e-3, 1, SEED_TRAIN)
    bd, dc = Boundary.from_coords(s["det_coords"]), s["det_coords"]
    pattern = [(6, 6, 3), (8, 8, 3), (8, 8, 4)]
    for dx, dy, dt in [(4, 0, 0), (0, 4, 0), (2, 2, 1), (4, 4, 1)]:
        moved = _cluster_at(dc, [(x + dx, y + dy, t + dt) for x, y, t in pattern])
        assert canonical_key(moved, bd) == canonical_key(_cluster_at(dc, pattern), bd)


def test_key_ignores_detector_order(shots, graph):
    _, bd = graph
    c = _cluster_at(shots["train"]["det_coords"], [(4, 4, 2), (6, 6, 2)])
    rev = Cluster(det_ids=c.det_ids[::-1], coords=c.coords[::-1])
    assert canonical_key(c, bd) == canonical_key(rev, bd)


def test_different_shapes_different_keys(shots, graph):
    _, bd = graph
    dc = shots["train"]["det_coords"]
    a = _cluster_at(dc, [(4, 4, 2), (4, 4, 3)])
    b = _cluster_at(dc, [(4, 4, 2), (6, 6, 2)])
    assert canonical_key(a, bd) != canonical_key(b, bd)


def test_shift_that_swaps_stabilizer_type_changes_key(shots, graph):
    _, bd = graph
    dc = shots["train"]["det_coords"]
    z = _cluster_at(dc, [(4, 4, 2)])   # (x+y)/2 even -> Z-type
    x = _cluster_at(dc, [(6, 4, 2)])   # (x+y)/2 odd  -> X-type
    assert canonical_key(z, bd) != canonical_key(x, bd)


def test_near_edge_does_not_match_bulk(shots, graph):
    _, bd = graph
    dc = shots["train"]["det_coords"]
    bulk = _cluster_at(dc, [(4, 4, 2)])
    assert canonical_key(_cluster_at(dc, [(2, 2, 2)]), bd) != canonical_key(bulk, bd)  # space
    assert canonical_key(_cluster_at(dc, [(4, 4, 1)]), bd) != canonical_key(bulk, bd)  # time


def test_clamp_below_reach_rejected(graph):
    _, bd = graph
    with pytest.raises(ValueError):
        Boundary(lo=bd.lo, hi=bd.hi, step=bd.step, reach=3, clamp=2)


# ---------------------------------------------------------------- isolation
def test_lone_cluster_is_isolated(shots, graph):
    _, bd = graph
    c = _cluster_at(shots["train"]["det_coords"], [(4, 4, 2), (4, 4, 3)])
    assert is_isolated(c, [c], bd)


def test_touching_clusters_not_isolated(shots, graph):
    _, bd = graph
    dc = shots["train"]["det_coords"]
    a = _cluster_at(dc, [(4, 4, 2)])
    b = _cluster_at(dc, [(4, 4, 4)])          # 2 steps apart in time, reach is 2
    assert not is_isolated(a, [a, b], bd)
    assert not is_isolated(b, [a, b], bd)


def test_far_clusters_isolated(shots, graph):
    _, bd = graph
    dc = shots["train"]["det_coords"]
    a = _cluster_at(dc, [(2, 2, 1)])
    b = _cluster_at(dc, [(8, 8, 4)])          # 3 steps apart in every axis
    assert is_isolated(a, [a, b], bd) and is_isolated(b, [a, b], bd)


def test_real_shots_clusters_not_adjacent(shots, graph):
    """find_clusters separates clusters with no shared fault; isolation is a stricter gap."""
    A, _ = graph
    s = shots["test"]
    for row in s["dets"][:300]:
        cl = find_clusters(row, s["det_coords"], A)
        for i, a in enumerate(cl):
            for b in cl[i + 1:]:
                assert A[list(a.det_ids)][:, list(b.det_ids)].nnz == 0
