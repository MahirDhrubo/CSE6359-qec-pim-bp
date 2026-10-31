# OWNER: P2
"""Split a shot's detection events into connected clusters of nearby defects."""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class Cluster:
    """One connected group of detection events. P2 may add fields as needed."""
    det_ids: tuple[int, ...]   # detector indices (sorted)
    coords: np.ndarray         # float[len(det_ids) x k], rows of det_coords


def detector_adjacency(H: sp.spmatrix) -> sp.csr_matrix:
    """Detector graph from the decoding matrix: two detectors are adjacent iff some fault
    column of H flips both of them.

    Inputs:  H csr [n_checks x n_vars] from GRAPHS_DIR/<code>_d<d>_p<p>.npz (checks == detectors).
    Outputs: symmetric bool csr [D x D] with an empty diagonal.
    Owner:   P2
    """
    Hb = sp.csr_matrix(H, dtype=np.int32)
    Hb.data[:] = 1
    A = (Hb @ Hb.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1
    return A.astype(bool)


def find_clusters(
    det_events: np.ndarray, det_coords: np.ndarray, adjacency: sp.csr_matrix
) -> list[Cluster]:
    """Group the fired detectors of one shot into connected components.

    Inputs:  det_events bool[D] (one row of SHOTS_DIR/<tag>.npz "dets"),
             det_coords float[D x k] (same npz), adjacency csr[D x D] detector graph
             (two detectors adjacent iff they share a fault column in H from GRAPHS_DIR).
    Outputs: list[Cluster], one per connected component of fired detectors.
    Contract: reads SHOTS_DIR and GRAPHS_DIR npz files (io.load_npz).
    Owner:   P2
    """
    fired = np.flatnonzero(det_events)
    if fired.size == 0:
        return []
    sub = adjacency[fired][:, fired]
    _, labels = connected_components(sub, directed=False)
    order = np.argsort(labels, kind="stable")       # fired is sorted, so ids stay sorted
    splits = np.flatnonzero(np.diff(labels[order])) + 1
    return [
        Cluster(det_ids=tuple(int(i) for i in fired[idx]), coords=det_coords[fired[idx]])
        for idx in np.split(order, splits)
    ]
