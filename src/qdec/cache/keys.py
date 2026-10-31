# OWNER: P2
"""Translation-invariant cache keys for clusters, and the isolation check that makes a hit safe.

Geometry is measured in lattice steps (2 coordinate units in space, 1 round in time for Stim's
rotated surface code), not graph hops: the undecomposed DEM has hyperedges, so every detector is
within 2 hops of most of a d=5 patch and hop distance cannot tell near from far.
"""
from dataclasses import dataclass

import numpy as np

from qdec.cache.splitter import Cluster

KEY_VERSION = 1


@dataclass(frozen=True)
class Boundary:
    """Patch geometry that keys and the isolation check need.

    lo, hi: float[k] coordinate extent of the detector lattice (spatial edges + first/last round).
    step:   float[k] lattice spacing per axis.
    reach:  steps (Chebyshev) within which another cluster may change the correct correction.
    clamp:  edge distances up to this many steps go into the key; beyond it the cluster is bulk.
    """
    lo: np.ndarray
    hi: np.ndarray
    step: np.ndarray
    reach: int = 2
    clamp: int = 2

    def __post_init__(self):
        if self.clamp < self.reach:
            raise ValueError("clamp must be >= reach, or a key could hide a nearby edge")

    @classmethod
    def from_coords(cls, det_coords: np.ndarray, reach: int = 2, clamp: int = 2) -> "Boundary":
        """Infer extents and lattice spacing from SHOTS_DIR/<tag>.npz det_coords."""
        lo, hi = det_coords.min(axis=0), det_coords.max(axis=0)
        step = np.ones(det_coords.shape[1])
        for a in range(det_coords.shape[1]):
            gaps = np.diff(np.unique(det_coords[:, a]))
            if gaps.size:
                step[a] = gaps.min()
        return cls(lo=lo, hi=hi, step=step, reach=reach, clamp=clamp)

    def grid(self, coords: np.ndarray) -> np.ndarray:
        """Coordinates -> int lattice indices, origin at the low corner of the patch."""
        return np.rint((np.asarray(coords, dtype=np.float64) - self.lo) / self.step).astype(np.int64)


def canonical_key(cluster: Cluster, boundary: Boundary) -> bytes:
    """Map a cluster to a key that is identical for every translate of the same defect pattern.

    Key = (detector offsets from the lexicographically first detector, sorted)
        + sublattice class of that anchor ((x + y) % 2 in steps: a (1,0) shift swaps X- and
          Z-type stabilizers, (1,1) and (2,0) shifts preserve type and CNOT schedule)
        + distance in steps to each patch edge, clamped at boundary.clamp, so clusters near an
          edge or the first/last round only match clusters with the same surroundings.

    Inputs:  Cluster from splitter.find_clusters, Boundary of the same patch.
    Outputs: bytes key for the CAM.
    Contract: none on disk; used by cam.ClusterCache.
    Owner:   P2
    """
    g = boundary.grid(cluster.coords)
    g = g[np.lexsort(g.T[::-1])]
    anchor = g[0]
    span = boundary.grid(boundary.hi)
    edges = np.minimum(np.concatenate([g.min(axis=0), span - g.max(axis=0)]), boundary.clamp)
    sublattice = int(anchor[0] + anchor[1]) % 2 if g.shape[1] >= 2 else 0
    header = np.array([KEY_VERSION, len(g), g.shape[1], sublattice, *edges], dtype=np.int16)
    return header.tobytes() + (g - anchor).astype(np.int16).tobytes()


def is_isolated(cluster: Cluster, all_clusters: list[Cluster], boundary: Boundary) -> bool:
    """True iff the cached correction for `cluster` can be applied without interacting with any
    other cluster or the code boundary (otherwise the shot must go to BP).

    Edges are handled by the key (edge distances up to clamp >= reach are part of it), so the
    check here is: no detector of another cluster within `reach` steps (Chebyshev) of this one.

    Inputs:  the cluster, every cluster of the same shot, a boundary description of the patch.
    Outputs: bool. A shot is a cache hit only if ALL its clusters hit AND pass this check.
    Contract: feeds CACHE_COLS.cache_hit.
    Owner:   P2
    """
    own = set(cluster.det_ids)
    others = [
        c for o in all_clusters for d, c in zip(o.det_ids, o.coords) if d not in own
    ]
    if not others:
        return True
    mine = boundary.grid(cluster.coords)
    theirs = boundary.grid(np.array(others))
    gap = np.abs(mine[:, None, :] - theirs[None, :, :]).max(axis=2)
    return bool(gap.min() > boundary.reach)
