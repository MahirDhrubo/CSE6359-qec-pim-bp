# OWNER: P1
"""Detector error model -> parity-check decoding graph (H, L, priors) and its size statistics."""
import numpy as np
import scipy.sparse as sp
import stim

from qdec.contracts import GRAPH_STATS_KEYS  # noqa: F401  (keys graph_stats must return)


def dem_to_graph(circuit: stim.Circuit) -> dict:
    """Convert circuit.detector_error_model(decompose_errors=False) into a BP decoding graph.

    Inputs:  stim.Circuit from circuits.make_circuit.
    Outputs: {"H": csr bool/uint8 [n_checks x n_vars]   detector-to-fault incidence,
              "L": csr bool/uint8 [n_obs x n_vars]      observable-to-fault incidence,
              "priors": float64[n_vars]                 per-fault probability}.
    Contract: written via io.save_npz to GRAPHS_DIR / f"{code}_d{d}_p{p:g}.npz".
    Owner:   P1
    """
    dem = circuit.detector_error_model(decompose_errors=False)
    g, _ = _merged_columns(dem)
    return g


def _merged_columns(dem: stim.DetectorErrorModel) -> tuple[dict, np.ndarray]:
    """Build H, L, priors with one column per distinct (detectors, observables) symptom.

    Flattening REPEAT blocks can emit the same symptom more than once (stim only merges within
    a block), so duplicates are merged as independent XOR events: p = p1 + p2 - 2 p1 p2.
    Also returns col_of[i] = column of the i-th `error` instruction of dem.flattened(), so an
    error vector from dem.compile_sampler() maps onto the columns (used by the tests).
    """
    col_index: dict[tuple, int] = {}
    symptoms, priors, col_of = [], [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        ts = inst.targets_copy()
        key = (
            tuple(sorted(t.val for t in ts if t.is_relative_detector_id())),
            tuple(sorted(t.val for t in ts if t.is_logical_observable_id())),
        )
        j = col_index.get(key)
        if j is None:
            j = col_index[key] = len(priors)
            symptoms.append(key)
            priors.append(p)
        else:
            priors[j] = priors[j] + p - 2 * priors[j] * p
        col_of.append(j)
    n = len(priors)

    def incidence(which, m):
        rows = [r for s in symptoms for r in s[which]]
        cols = [j for j, s in enumerate(symptoms) for _ in s[which]]
        return sp.csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(m, n))

    g = {
        "H": incidence(0, dem.num_detectors),
        "L": incidence(1, dem.num_observables),
        "priors": np.asarray(priors, dtype=np.float64),
    }
    return g, np.asarray(col_of, dtype=np.int64)


def graph_stats(H: sp.csr_matrix) -> dict[str, int]:
    """Size statistics the cost model needs.

    Inputs:  H from dem_to_graph.
    Outputs: dict with exactly contracts.GRAPH_STATS_KEYS:
             n_vars, n_checks, n_edges (nnz of H), max_var_deg, max_check_deg.
    Contract: written as JSON to GRAPHS_DIR / f"{code}_d{d}_p{p:g}.json"; read by P2 cost model.
    Owner:   P1
    """
    raise NotImplementedError
