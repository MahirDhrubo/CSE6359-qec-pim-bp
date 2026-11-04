# OWNER: P2
"""Day 2: ClusterCache build (TRAIN only) / evaluate (TEST) on P2's own Stim shots."""
import numpy as np
import pandas as pd
import pytest

from qdec.cache.cam import ClusterCache, Entry
from qdec.cache.devshots import sample_dev_shots
from qdec.cache.keys import canonical_key, is_isolated
from qdec.cache.splitter import Cluster
from qdec.contracts import CACHE_COLS, SEED_TEST, SEED_TRAIN

TRAIN_TAG, TEST_TAG = f"surface_d5_p0.001_s{SEED_TRAIN}", f"surface_d5_p0.001_s{SEED_TEST}"


@pytest.fixture(scope="module")
def shots():
    return {"train": sample_dev_shots(5, 1e-3, 2000, SEED_TRAIN),
            "test": sample_dev_shots(5, 1e-3, 2000, SEED_TEST)}


@pytest.fixture(scope="module")
def built(shots):
    tr = shots["train"]
    cache = ClusterCache(tr, tr["det_coords"])
    cache.build(tr, TRAIN_TAG)
    splits = cache.split(shots["test"])
    return cache, splits, cache.evaluate(shots["test"], TEST_TAG, splits)


# ---------------------------------------------------------------- TRAIN / TEST separation
def test_build_refuses_test_seed(shots):
    te = shots["test"]
    with pytest.raises(ValueError, match="SEED_TRAIN"):
        ClusterCache(te, te["det_coords"]).build(te, TEST_TAG)


def test_evaluate_refuses_train_tag_and_train_shots(built, shots):
    cache, _, _ = built
    with pytest.raises(ValueError):
        cache.evaluate(shots["test"], TRAIN_TAG)
    with pytest.raises(ValueError, match="identical"):
        cache.evaluate(shots["train"], TEST_TAG)      # TRAIN shots relabeled as TEST


def test_evaluate_before_build_raises(shots):
    tr = shots["train"]
    with pytest.raises(RuntimeError):
        ClusterCache(tr, tr["det_coords"]).evaluate(shots["test"], TEST_TAG)


# ---------------------------------------------------------------- output contract
def test_evaluate_schema_and_null_on_miss(built, shots):
    _, splits, df = built
    pd.testing.assert_frame_equal(df, df.astype(CACHE_COLS))
    assert len(df) == len(shots["test"]["dets"])
    assert (df["shot_id"] == np.arange(len(df))).all()
    assert df["fail_cache"].isna().equals(~df["cache_hit"])
    hit = df["cache_hit"].to_numpy()
    # a hit looked up every cluster, and every one of them was isolated
    assert (df["n_lookups"][hit] == df["n_clusters"][hit]).all()
    assert all(all(splits[i].isolated) for i in np.flatnonzero(hit))
    assert (df["n_lookups"] <= df["n_clusters"]).all()


def test_empty_shots_hit_without_lookups(built, shots):
    _, _, df = built
    empty = ~shots["test"]["dets"].any(axis=1)
    assert empty.any()
    assert df["cache_hit"][empty].all() and (df["n_lookups"][empty] == 0).all()


def test_hit_rate_and_ler_sane(built):
    cache, _, df = built
    nonempty = df["n_clusters"] > 0
    assert cache.size() > 0 and cache.key_bits() > 0
    assert 0.5 < df["cache_hit"][nonempty].mean() < 1.0      # d=5 p=1e-3: most shots hit
    assert df["fail_cache"].sum() <= 1                        # local corrections are ~never wrong


def test_isolation_flags_match_is_isolated(built):
    cache, splits, _ = built
    multi = [s for s in splits if len(s.clusters) > 1]
    assert len(multi) > 50
    for s in multi:
        assert s.isolated == [is_isolated(c, s.clusters, cache.boundary) for c in s.clusters]


def test_cluster_status_agrees_with_evaluate(built):
    cache, splits, df = built
    for s, hit in zip(splits, df["cache_hit"]):
        st = cache.cluster_status(s)
        assert len(st) == len(s.clusters)
        assert set(st) <= {"hit", "CAM miss", "not isolated", "too big", "no fit"}
        assert hit == all(x == "hit" for x in st)


def test_capped_build_keeps_most_frequent_keys(built, shots):
    full, _, _ = built
    tr = shots["train"]
    capped = ClusterCache(tr, tr["det_coords"], max_entries=20)
    capped.build(tr, TRAIN_TAG)
    assert capped.size() <= 20 and len(capped.entries) <= 20
    top = sorted(full.entries.items(), key=lambda kv: (-kv[1].count, kv[0]))[:capped.size()]
    assert set(capped.entries) == {k for k, _ in top}


def test_resize_is_monotone(built, shots):
    cache, splits, _ = built
    rates = []
    for n in (1, 10, 100, None):
        cache.resize(n)
        rates.append(cache.evaluate(shots["test"], TEST_TAG, splits)["cache_hit"].mean())
        assert cache.size() == (len(cache.entries) if n is None else min(n, len(cache.entries)))
    assert rates == sorted(rates)
    cache.resize(None)


# ---------------------------------------------------------------- correction replay
def _single_fault_cluster(cache, j):
    ids = tuple(cache._Hc.indices[cache._Hc.indptr[j]:cache._Hc.indptr[j + 1]].tolist())
    return Cluster(det_ids=ids, coords=cache.det_coords[list(ids)])


def test_correction_replays_at_translate():
    """A fault learned at one place replays as the translated fault, with that fault's own
    observable flip (the logical effect is position dependent, the shape is not).
    Every pair whose keys match (the cache's own replay condition) is checked, including shifts
    along an edge, where the memory-Z observable lives."""
    g = sample_dev_shots(9, 1e-3, 1, SEED_TRAIN)
    cache = ClusterCache(g, g["det_coords"])
    checked = flips_seen = 0
    for shift in ([2, 0, 0], [0, 2, 0], [1, 1, 0], [0, 0, 1]):   # lattice steps, sublattice kept
        for dets, j in list(cache._col.items()):
            moved = [cache._det_at.get(tuple(cache._grid[d] + shift)) for d in dets]
            if None in moved or (j2 := cache._col.get(tuple(sorted(moved)))) is None:
                continue
            a, b = _single_fault_cluster(cache, j), _single_fault_cluster(cache, j2)
            if canonical_key(a, cache.boundary) != canonical_key(b, cache.boundary):
                continue
            flips = cache._replay(Entry(cache._local_correction(a)), b)
            assert flips is not None
            np.testing.assert_array_equal(flips, cache._fault_flips[j2])
            checked += 1
            flips_seen += bool(flips.any())
    assert checked > 500 and flips_seen > 10
