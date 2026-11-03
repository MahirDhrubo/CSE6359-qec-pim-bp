# OWNER: P2
"""Dev-only Stim shots so P2 can work before P1's SHOTS_DIR / GRAPHS_DIR files exist.

Returns the same keys as P1's shots npz (dets, obs, det_coords) plus H, L, priors from the
detector error model. Swap to io.load_npz on P1's files once they land; nothing else changes.
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


def dev_graph(circuit: stim.Circuit) -> dict:
    """{"H", "L", "priors"} like GRAPHS_DIR npz: one column per distinct (detectors, observables)
    symptom of the undecomposed DEM, duplicates merged as independent XOR events."""
    index, symptoms, priors = {}, [], []
    for inst in circuit.detector_error_model(decompose_errors=False).flattened():
        if inst.type != "error":
            continue
        ts, p = inst.targets_copy(), inst.args_copy()[0]
        key = (tuple(sorted(t.val for t in ts if t.is_relative_detector_id())),
               tuple(sorted(t.val for t in ts if t.is_logical_observable_id())))
        if not key[0]:
            continue
        j = index.setdefault(key, len(priors))
        if j == len(priors):
            symptoms.append(key)
            priors.append(p)
        else:
            priors[j] = priors[j] + p - 2 * priors[j] * p

    def incidence(which, m):
        rows = [r for s in symptoms for r in s[which]]
        cols = [j for j, s in enumerate(symptoms) for _ in s[which]]
        return sp.csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(m, len(priors)))

    return {"H": incidence(0, circuit.num_detectors), "L": incidence(1, circuit.num_observables),
            "priors": np.asarray(priors, dtype=np.float64)}


def dem_matrix(circuit: stim.Circuit) -> sp.csr_matrix:
    """Detector-to-fault incidence H [D x n_faults] from the undecomposed detector error model."""
    return dev_graph(circuit)["H"]


def sample_dev_shots(distance: int, p: float, shots: int, seed: int) -> dict:
    """{"dets": bool[S x D], "obs": bool[S x O], "det_coords": float[D x k], "H", "L": csr,
    "priors": float[n_faults]}."""
    c = surface_circuit(distance, p)
    dets, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    coords = c.get_detector_coordinates()
    det_coords = np.array([coords[i] for i in range(c.num_detectors)], dtype=np.float64)
    return {"dets": dets, "obs": obs, "det_coords": det_coords, **dev_graph(c)}
