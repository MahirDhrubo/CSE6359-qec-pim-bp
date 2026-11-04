# OWNER: P1
import dataclasses

import numpy as np
import pytest

from qdec.contracts import QUICK_GRID, SEED_TEST, SEED_TRAIN, Workload
from qdec.workload.circuits import make_circuit, sample_shots


@pytest.mark.parametrize("w", QUICK_GRID, ids=lambda w: w.tag)
def test_surface_circuit_shape(w):
    c = make_circuit(w)
    d, r = w.distance, w.rounds
    # rotated memory_z with s = (d^2-1)/2 stabilizers of each type: Z detectors in all r rounds
    # plus the final data-readout round, X detectors only in rounds 2..r (round 1 is random).
    s = (d * d - 1) // 2
    assert c.num_detectors == s * (r + 1) + s * (r - 1)
    assert c.num_observables == 1
    # stim indexes qubits sparsely (num_qubits = max index + 1), so count the placed qubits
    assert len(c.get_final_qubit_coordinates()) == 2 * d * d - 1


def test_every_detector_has_coordinates():
    c = make_circuit(QUICK_GRID[0])
    coords = c.get_detector_coordinates()
    assert len(coords) == c.num_detectors
    assert all(len(v) >= 2 for v in coords.values())


def test_noise_is_applied_at_w_p():
    w = QUICK_GRID[0]
    c = make_circuit(w)
    probs = {inst.gate_args_copy()[0] for inst in c.flattened() if inst.name in
             ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR")}
    assert probs == {w.p}
    noiseless = make_circuit(dataclasses.replace(w, p=0.0))
    assert noiseless.detector_error_model().num_errors == 0


def test_deterministic():
    w = QUICK_GRID[0]
    assert make_circuit(w) == make_circuit(w)


def test_gross_memory_shape():
    c = make_circuit(Workload("gross", 12, 1e-3, 12, 10, 0))
    assert c.num_qubits == 288 and c.num_observables == 12
    assert c.num_detectors == 144 * 12


def test_unknown_code_rejected():
    with pytest.raises(ValueError):
        make_circuit(Workload("color", 5, 1e-3, 5, 10, 0))


def _small(w, shots=2_000):
    return dataclasses.replace(w, shots=shots)


@pytest.mark.parametrize("w", QUICK_GRID, ids=lambda w: w.tag)
def test_sample_shots_shapes_and_dtypes(w):
    w = _small(w, 500)
    c = make_circuit(w)
    s = sample_shots(w)
    assert set(s) == {"dets", "obs", "det_coords"}
    assert s["dets"].shape == (w.shots, c.num_detectors) and s["dets"].dtype == bool
    assert s["obs"].shape == (w.shots, c.num_observables) and s["obs"].dtype == bool
    assert s["det_coords"].shape[0] == c.num_detectors
    assert s["det_coords"].dtype == np.float64 and not np.isnan(s["det_coords"]).any()


def test_det_coords_match_circuit():
    w = _small(QUICK_GRID[0], 10)
    coords = make_circuit(w).get_detector_coordinates()
    got = sample_shots(w)["det_coords"]
    for i, v in coords.items():
        np.testing.assert_array_equal(got[i], v)


def test_same_seed_same_shots_and_train_differs_from_test():
    w = _small(QUICK_GRID[0])
    a, b = sample_shots(w), sample_shots(w)
    np.testing.assert_array_equal(a["dets"], b["dets"])
    np.testing.assert_array_equal(a["obs"], b["obs"])
    train = sample_shots(dataclasses.replace(w, seed=SEED_TRAIN))
    assert w.seed == SEED_TEST
    assert not np.array_equal(a["dets"], train["dets"])


def test_defect_rate_grows_with_p():
    lo, hi = (_small(w) for w in QUICK_GRID[:2])
    assert lo.distance == hi.distance and lo.p < hi.p
    assert sample_shots(lo)["dets"].mean() < sample_shots(hi)["dets"].mean()


def test_noiseless_shots_are_trivial():
    s = sample_shots(dataclasses.replace(_small(QUICK_GRID[0], 200), p=0.0))
    assert not s["dets"].any() and not s["obs"].any()
