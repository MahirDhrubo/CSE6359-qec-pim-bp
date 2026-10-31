# OWNER: P1
"""Reference decoders: ldpc (v2) BP+OSD and PyMatching, with measured CPU wall-clock latency."""
import numpy as np
import scipy.sparse as sp


def bposd_decode_batch(
    H: sp.csr_matrix,
    L: sp.csr_matrix,
    priors: np.ndarray,
    dets: np.ndarray,
    obs: np.ndarray,
    max_iters: int,
    osd_order: int,
) -> dict[str, np.ndarray]:
    """Decode every shot with ldpc.BpOsdDecoder and time each decode.

    Inputs:  graph (H, L, priors) from GRAPHS_DIR; dets bool[S x D], obs bool[S x O] from SHOTS_DIR.
    Outputs: {"fail": bool[S], "bp_iters": int[S], "bp_converged": bool[S], "osd_ran": bool[S],
              "latency_ns": float[S]  (MEASURED wall clock, time.perf_counter_ns per shot)}.
    Contract: fail -> FUNCTIONAL_COLS.fail_bp_float; latency_ns -> P2's bposd_cpu service rows.
    Owner:   P1
    """
    raise NotImplementedError


def matching_decode_batch(circuit, dets: np.ndarray, obs: np.ndarray) -> dict[str, np.ndarray]:
    """Decode every shot with PyMatching (surface code only) and time each decode.

    Inputs:  stim.Circuit (for its detector error model); dets, obs as above.
    Outputs: {"fail": bool[S], "latency_ns": float[S] (MEASURED wall clock)}.
    Contract: fail -> FUNCTIONAL_COLS.fail_matching (null for the gross code).
    Owner:   P1
    """
    raise NotImplementedError
