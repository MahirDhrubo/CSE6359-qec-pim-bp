# OWNER: P2
"""Dev-only Stim shots so P2 can work before P1's SHOTS_DIR / GRAPHS_DIR files exist.

Returns the same keys as P1's shots npz (dets, obs, det_coords) plus H from the detector
error model. Swap to io.load_npz on P1's files once they land; nothing else changes.
"""
import numpy as np
import scipy.sparse as sp
import stim


def surface_circuit(distance: int, p: float, rounds: int | None = None) -> stim.Circuit:
    """Rotated surface-code Z-memory with uniform circuit-level depolarizing noise at p."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds or distance,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def dem_matrix(circuit: stim.Circuit) -> sp.csr_matrix:
    """Detector-to-fault incidence H [D x n_faults] from the undecomposed detector error model."""
    rows, cols, n = [], [], 0
    for inst in circuit.detector_error_model(decompose_errors=False).flattened():
        if inst.type != "error":
            continue
        dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        if dets:
            rows += dets
            cols += [n] * len(dets)
            n += 1
    data = np.ones(len(rows), dtype=np.uint8)
    return sp.csr_matrix((data, (rows, cols)), shape=(circuit.num_detectors, n))


def sample_dev_shots(distance: int, p: float, shots: int, seed: int) -> dict:
    """{"dets": bool[S x D], "obs": bool[S x O], "det_coords": float[D x k], "H": csr}."""
    c = surface_circuit(distance, p)
    dets, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    coords = c.get_detector_coordinates()
    det_coords = np.array([coords[i] for i in range(c.num_detectors)], dtype=np.float64)
    return {"dets": dets, "obs": obs, "det_coords": det_coords, "H": dem_matrix(c)}
