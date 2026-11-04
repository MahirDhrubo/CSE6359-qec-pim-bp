# OWNER: P2
"""Content-addressable cluster cache (CAM model): canonical key -> cached local correction.

Build (SEED_TRAIN only): count the key of every TRAIN cluster. Each key's correction is found
once by decoding that cluster alone (ldpc BP+OSD restricted to the faults within `reach` of it)
and stored as fault shapes relative to the cluster's anchor detector, so it replays at any
translate. TRAIN shots made of a single cluster then replay the stored correction against the
true observable; a key whose correction is wrong in most of them is dropped.

Evaluate (SEED_TEST): a shot hits iff every cluster is isolated, has a key in the CAM, and its
replayed correction reproduces exactly that cluster's detectors. fail_cache compares the
observable flips of the replayed corrections with the shot's real observables.
"""
import hashlib
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from ldpc import BpOsdDecoder

from qdec.cache.keys import Boundary, canonical_key, isolation_flags, sorted_grid
from qdec.cache.splitter import Cluster, detector_adjacency, find_clusters
from qdec.contracts import CACHE_COLS, SEED_TEST, SEED_TRAIN

# Build-time local decoder (runs once per key on TRAIN, never on the test path).
LOCAL_MAX_ITERS = 50
LOCAL_OSD_ORDER = 10
# CAM design point: on QUICK_GRID every tag's hit rate saturates by 16384 entries
# (data/p2/cache_sweep.csv), and the cap bounds build time on FULL_GRID.
DEFAULT_CAM_ENTRIES = 16384
# Codes the cache is built for. Keys rely on the surface code's lattice translation symmetry,
# which is untested for the gross code, so gross runs PIM-BP without a cache (P2 guide fallback).
CACHEABLE_CODES = ("surface",)


@dataclass
class Entry:
    """One CAM entry: fault shapes (detector offsets from the anchor) plus TRAIN statistics."""
    faults: tuple[tuple[tuple[int, ...], ...], ...]
    count: int = 0      # TRAIN clusters with this key (CAM keeps the most frequent)
    checked: int = 0    # single-cluster TRAIN shots replayed with this entry
    wrong: int = 0      # ... in which the replayed correction got the observable wrong


@dataclass
class ShotSplit:
    """A shot's clusters with their keys (None if larger than max_cluster_size) and isolation."""
    clusters: list[Cluster]
    keys: list[bytes | None]
    isolated: list[bool]


def _fingerprint(dets: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(dets, dtype=bool)).tobytes()).hexdigest()


class ClusterCache:
    """Cache of cluster corrections built from SEED_TRAIN shots only.

    Inputs:  graph = io.load_npz(GRAPHS_DIR/<code>_d<d>_p<p>.npz) (H, L, priors),
             det_coords from the matching SHOTS_DIR npz,
             max_entries = CAM capacity (most frequent TRAIN keys kept; None = all),
             reach = isolation distance in lattice steps, max_cluster_size = largest cluster
             that gets a key (bigger clusters always miss).
    Owner: P2
    """

    def __init__(self, graph: dict, det_coords: np.ndarray, max_entries: int | None = None,
                 reach: int = 2, max_cluster_size: int = 8):
        self.H = sp.csr_matrix(graph["H"], dtype=np.uint8)
        self.priors = np.asarray(graph["priors"], dtype=np.float64)
        self.det_coords = np.asarray(det_coords, dtype=np.float64)
        if self.H.shape[0] != len(self.det_coords):
            raise ValueError("graph H rows and det_coords disagree on the number of detectors")
        self.boundary = Boundary.from_coords(self.det_coords, reach=reach, clamp=max(2, reach))
        self.adjacency = detector_adjacency(self.H)
        self.max_entries = max_entries
        self.max_cluster_size = max_cluster_size

        self._grid = self.boundary.grid(self.det_coords)
        self._det_at = {tuple(g): i for i, g in enumerate(self._grid.tolist())}
        self._fault_flips = sp.csr_matrix(graph["L"], dtype=np.uint8).T.toarray().astype(bool)
        # detector set -> fault column; on duplicates (same detectors, different observables)
        # keep the most likely fault, as a decoder would
        self._Hc = Hc = self.H.tocsc()
        self._col: dict[tuple[int, ...], int] = {}
        for j in np.argsort(-self.priors, kind="stable"):
            dets = tuple(Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]].tolist())
            self._col.setdefault(dets, int(j))

        self.entries: dict[bytes, Entry] = {}
        self._active: dict[bytes, Entry] = {}
        self._train_fp: str | None = None
        self.n_dropped = 0

    # ------------------------------------------------------------------ split
    def iter_split(self, shots: dict):
        """Clusters, keys and isolation flags, one ShotSplit per shot, computed lazily."""
        if not np.array_equal(np.asarray(shots["det_coords"]), self.det_coords, equal_nan=True):
            raise ValueError("shots were sampled from a different circuit than this cache's graph")
        for row in np.asarray(shots["dets"], dtype=bool):
            cl = find_clusters(row, self.det_coords, self.adjacency)
            keys = [canonical_key(c, self.boundary) if len(c.det_ids) <= self.max_cluster_size
                    else None for c in cl]
            yield ShotSplit(cl, keys, isolation_flags(cl, self.boundary))

    def split(self, shots: dict) -> list[ShotSplit]:
        """All of iter_split in memory (reusable across max_entries; QUICK_GRID sizes only)."""
        return list(self.iter_split(shots))

    # ------------------------------------------------------------------ build
    def build(self, train_shots: dict, tag: str, splits: list[ShotSplit] | None = None) -> None:
        """Populate the cache from training shots.

        Inputs:  dict from io.load_npz(SHOTS_DIR / f"{tag}.npz") for a SEED_TRAIN workload
                 (keys dets, obs, det_coords) and that workload's tag.
        Outputs: none (fills the CAM). Raises ValueError for a SEED_TEST tag.
                 Keys are decoded most frequent first and only until max_entries are stored,
                 so resize() can shrink the CAM afterwards but not grow it past max_entries.
        """
        if not tag.endswith(f"_s{SEED_TRAIN}"):
            raise ValueError(f"cache must be built on SEED_TRAIN shots only, got {tag!r}")
        self._train_fp = _fingerprint(train_shots["dets"])

        counts: Counter[bytes] = Counter()
        example: dict[bytes, Cluster] = {}
        singles = []    # (key, cluster, obs) of single-cluster TRAIN shots, for the replay check
        obs = np.asarray(train_shots["obs"], dtype=bool)
        for s, o in zip(splits if splits is not None else self.iter_split(train_shots), obs):
            for c, k in zip(s.clusters, s.keys):
                if k is not None:
                    counts[k] += 1
                    example.setdefault(k, c)
            if len(s.clusters) == 1 and s.keys[0] is not None:
                singles.append((s.keys[0], s.clusters[0], o))
        self.entries = {}
        for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if self.max_entries is not None and len(self.entries) >= self.max_entries:
                break
            faults = self._local_correction(example[k])
            if faults is not None:
                self.entries[k] = Entry(faults, count=n)

        for k, c, o in singles:
            if (e := self.entries.get(k)) is not None and (flips := self._replay(e, c)) is not None:
                e.checked += 1
                e.wrong += bool((flips != o).any())
        bad = [k for k, e in self.entries.items() if 2 * e.wrong > e.checked]
        for k in bad:
            del self.entries[k]
        self.n_dropped = len(bad)
        self.resize(self.max_entries)

    def resize(self, max_entries: int | None) -> None:
        """Keep the `max_entries` most frequent TRAIN keys active (None = all)."""
        self.max_entries = max_entries
        ranked = sorted(self.entries.items(), key=lambda kv: (-kv[1].count, kv[0]))
        self._active = dict(ranked[:max_entries] if max_entries is not None else ranked)

    def _local_correction(self, cl: Cluster):
        """Decode one cluster on the faults within `reach` of it; fault shapes or None.
        A cluster that one fault explains exactly gets that (most likely) fault directly."""
        single = self._col.get(cl.det_ids)
        if single is not None:
            return self._shapes([single], cl)
        g = self.boundary.grid(cl.coords)
        near = (np.abs(self._grid[:, None, :] - g[None]).max(axis=2) <= self.boundary.reach).any(axis=1)
        near_ids = np.flatnonzero(near)
        cols = np.flatnonzero((self.H[~near].getnnz(axis=0) == 0) & (self.H[near].getnnz(axis=0) > 0))
        sub = self.H[near_ids][:, cols]
        syn = np.zeros(len(near_ids), dtype=np.uint8)
        syn[np.searchsorted(near_ids, cl.det_ids)] = 1
        # omp_thread_count=1: ldpc's threaded path crashed (access violation) under CPU load on
        # FULL_GRID d13; one thread is also faster on these small local matrices
        dec = BpOsdDecoder(sub, error_channel=list(self.priors[cols]), max_iter=LOCAL_MAX_ITERS,
                           bp_method="product_sum", osd_method="osd_cs",
                           osd_order=min(LOCAL_OSD_ORDER, max(0, len(cols) - len(near_ids))),
                           omp_thread_count=1)
        e = np.asarray(dec.decode(syn), dtype=np.uint8)
        if not np.array_equal(sub @ e % 2, syn):
            return None
        return self._shapes(cols[np.flatnonzero(e)], cl)

    def _shapes(self, fault_cols, cl: Cluster) -> tuple:
        """Fault columns -> their detectors' lattice offsets from the cluster's anchor."""
        anchor = sorted_grid(cl, self.boundary)[0]
        Hc = self._Hc
        return tuple(sorted(
            tuple(map(tuple, (self._grid[Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]]] - anchor).tolist()))
            for j in fault_cols
        ))

    def _replay(self, entry: Entry, cl: Cluster) -> np.ndarray | None:
        """Translate the entry's faults onto `cl`; observable flips, or None if they don't fit."""
        anchor = sorted_grid(cl, self.boundary)[0]
        flips = np.zeros(self._fault_flips.shape[1], dtype=bool)
        fired: set[int] = set()
        for fault in entry.faults:
            ids = [self._det_at.get(tuple(anchor + off)) for off in map(np.array, fault)]
            j = None if None in ids else self._col.get(tuple(sorted(ids)))
            if j is None:
                return None
            fired.symmetric_difference_update(ids)
            flips ^= self._fault_flips[j]
        return flips if fired == set(cl.det_ids) else None

    # ------------------------------------------------------------------ query
    def lookup(self, key: bytes):
        """Return the cached correction for `key`, or None on a miss."""
        e = self._active.get(key)
        return None if e is None else e.faults

    def size(self) -> int:
        """Number of CAM entries (for the CAM-size vs hit-rate plot)."""
        return len(self._active)

    def cluster_status(self, split: ShotSplit) -> list[str]:
        """Why each cluster of one shot would or would not be served (demo / reporting):
        "hit", "CAM miss", "not isolated", "too big" (> max_cluster_size) or "no fit" (cached
        shape does not fit here). Unlike evaluate(), every cluster is checked, no early exit."""
        out = []
        for c, k, iso in zip(split.clusters, split.keys, split.isolated):
            if k is None:
                out.append("too big")
            elif not iso:
                out.append("not isolated")
            elif (e := self._active.get(k)) is None:
                out.append("CAM miss")
            else:
                out.append("hit" if self._replay(e, c) is not None else "no fit")
        return out

    def key_bits(self) -> int:
        """CAM word width: the longest active key, in bits (search energy scales with it)."""
        return 8 * max((len(k) for k in self._active), default=0)

    def evaluate(self, test_shots: dict, tag: str,
                 splits: list[ShotSplit] | None = None) -> pd.DataFrame:
        """Run every SEED_TEST shot through the cache path.

        Inputs:  dict from io.load_npz(SHOTS_DIR / f"{tag}.npz") for the SEED_TEST workload.
        Outputs: DataFrame matching contracts.CACHE_COLS, one row per shot
                 (fail_cache is null on a miss). Clusters are checked in order and the shot
                 stops at the first non-isolated / uncacheable / missing cluster, so
                 n_lookups counts the CAM searches actually issued.
        Contract: written via io.write_table to TRACES_DIR / f"cache_{tag}.parquet".
        """
        if not tag.endswith(f"_s{SEED_TEST}"):
            raise ValueError(f"evaluate on SEED_TEST shots only, got {tag!r}")
        if self._train_fp is None:
            raise RuntimeError("build() the cache before evaluate()")
        if _fingerprint(test_shots["dets"]) == self._train_fp:
            raise ValueError("test shots are identical to the TRAIN shots the cache was built on")
        obs = np.asarray(test_shots["obs"], dtype=bool)
        n = len(obs)
        n_clusters, max_size, n_lookups = (np.zeros(n, dtype=np.int32) for _ in range(3))
        hit, fail = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        for i, s in enumerate(splits if splits is not None else self.iter_split(test_shots)):
            n_clusters[i] = len(s.clusters)
            max_size[i] = max((len(c.det_ids) for c in s.clusters), default=0)
            flips = np.zeros(obs.shape[1], dtype=bool)
            ok = True
            for c, k, iso in zip(s.clusters, s.keys, s.isolated):
                if not iso or k is None:
                    ok = False
                    break
                n_lookups[i] += 1
                e = self._active.get(k)
                f = None if e is None else self._replay(e, c)
                if f is None:
                    ok = False
                    break
                flips ^= f
            hit[i] = ok
            fail[i] = ok and bool((flips != obs[i]).any())
        df = pd.DataFrame({
            "shot_id": np.arange(n), "tag": tag, "n_clusters": n_clusters,
            "max_cluster_size": max_size, "n_lookups": n_lookups, "cache_hit": hit,
            "fail_cache": pd.array(np.where(hit, fail, None), dtype="boolean"),
        })
        return df.astype(CACHE_COLS)
