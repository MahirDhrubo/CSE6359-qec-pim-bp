# OWNER: P2
"""Split a shot's detection events into connected clusters of nearby defects."""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Cluster:
    """One connected group of detection events. P2 may add fields as needed."""
    det_ids: tuple[int, ...]   # detector indices (sorted)
    coords: np.ndarray         # float[len(det_ids) x k], rows of det_coords


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
    raise NotImplementedError
