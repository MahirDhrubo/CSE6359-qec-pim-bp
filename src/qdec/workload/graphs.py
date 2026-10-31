# OWNER: P1
"""Detector error model -> parity-check decoding graph (H, L, priors) and its size statistics."""
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
    raise NotImplementedError


def graph_stats(H: sp.csr_matrix) -> dict[str, int]:
    """Size statistics the cost model needs.

    Inputs:  H from dem_to_graph.
    Outputs: dict with exactly contracts.GRAPH_STATS_KEYS:
             n_vars, n_checks, n_edges (nnz of H), max_var_deg, max_check_deg.
    Contract: written as JSON to GRAPHS_DIR / f"{code}_d{d}_p{p:g}.json"; read by P2 cost model.
    Owner:   P1
    """
    raise NotImplementedError
