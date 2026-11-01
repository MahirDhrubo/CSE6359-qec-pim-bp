# OWNER: P1
"""Stim circuits and shot sampling for every Workload in contracts.QUICK_GRID / FULL_GRID."""
import numpy as np
import stim

from qdec.contracts import Workload


def make_circuit(w: Workload) -> stim.Circuit:
    """Build the noisy memory-experiment circuit for workload `w`.

    Inputs:  w.code == "surface" -> rotated surface-code memory experiment, distance w.distance,
             w.rounds syndrome rounds, circuit-level depolarizing noise at w.p
             (e.g. stim.Circuit.generated("surface_code:rotated_memory_z", ...)).
             w.code == "gross"   -> bivariate-bicycle [[144,12,12]] gross code memory experiment
             (w.distance == 12), w.rounds rounds, same noise model.
    Outputs: stim.Circuit with DETECTOR (with coordinates) and OBSERVABLE_INCLUDE annotations.
    Contract: none directly; feeds sample_shots() and graphs.dem_to_graph().
    Owner:   P1
    """
    if w.code == "surface":
        # Uniform circuit-level depolarizing noise: every Clifford, every data qubit before each
        # round, every reset and every measurement fail with probability w.p.
        return stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=w.distance,
            rounds=w.rounds,
            after_clifford_depolarization=w.p,
            before_round_data_depolarization=w.p,
            after_reset_flip_probability=w.p,
            before_measure_flip_probability=w.p,
        )
    if w.code == "gross":
        raise NotImplementedError("gross [[144,12,12]] circuit not built yet (P1 Day 2/3, T14)")
    raise ValueError(f"unknown code {w.code!r}; expected 'surface' or 'gross'")


def sample_shots(w: Workload) -> dict[str, np.ndarray]:
    """Sample w.shots shots from make_circuit(w) with seed w.seed.

    Inputs:  w (use w.seed; SEED_TRAIN shots build the cache, SEED_TEST shots evaluate everything).
    Outputs: {"dets": bool[S x D], "obs": bool[S x O], "det_coords": float[D x k]}.
    Contract: written by scripts/p1_make_data.py via io.save_npz to SHOTS_DIR / f"{w.tag}.npz".
    Owner:   P1
    """
    circuit = make_circuit(w)
    sampler = circuit.compile_detector_sampler(seed=w.seed)
    dets, obs = sampler.sample(w.shots, separate_observables=True)
    coords = circuit.get_detector_coordinates()
    k = max((len(v) for v in coords.values()), default=0)
    det_coords = np.full((circuit.num_detectors, k), np.nan)
    for i, v in coords.items():
        det_coords[i, : len(v)] = v
    return {"dets": dets.astype(bool), "obs": obs.astype(bool), "det_coords": det_coords}
