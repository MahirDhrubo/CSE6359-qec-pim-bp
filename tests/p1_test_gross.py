# OWNER: P1
from dataclasses import replace

import numpy as np
import pytest

from qdec.contracts import SEED_TEST, SEED_TRAIN, Workload
from qdec.workload.circuits import make_circuit, sample_shots
from qdec.workload.graphs import dem_to_graph
from qdec.workload.gross import rref_binary, structure


def test_gross_css_dimension_and_logical_quotient():
    hx, hz, lz, nx, nz = structure()
    assert hx.shape == hz.shape == (72, 144)
    assert (hx.sum(axis=1) == 6).all() and (hz.sum(axis=1) == 6).all()
    assert not (hx @ hz.T % 2).any()
    assert len(rref_binary(hx)[1]) == len(rref_binary(hz)[1]) == 66
    assert not (hx @ lz.T % 2).any()
    assert len(rref_binary(np.vstack([hz, lz]))[1]) == 78
    for checks, neighbors in ((hx, nx), (hz, nz)):
        for row, ns in zip(checks, neighbors, strict=True):
            assert set(np.flatnonzero(row)) == set(ns)


@pytest.mark.parametrize("rounds", [1, 2, 12])
def test_gross_noiseless_memory_and_deterministic_detectors(rounds):
    w = Workload("gross", 12, 0, rounds, 100, SEED_TEST)
    c = make_circuit(w)
    assert c.num_detectors == 144 * rounds and c.num_observables == 12
    assert c.detector_error_model().num_errors == 0
    s = sample_shots(w)
    assert not s["dets"].any() and not s["obs"].any()
    assert s["det_coords"].shape == (144 * rounds, 3)


def test_gross_noise_seed_and_graph():
    w = Workload("gross", 12, .001, 2, 64, SEED_TEST)
    a, b = sample_shots(w), sample_shots(w)
    train = sample_shots(replace(w, seed=SEED_TRAIN))
    np.testing.assert_array_equal(a["dets"], b["dets"])
    assert a["dets"].any() and a["obs"].shape == (64, 12)
    assert not np.array_equal(a["dets"], train["dets"])
    g = dem_to_graph(make_circuit(w))
    assert g["H"].shape[0] == 288 and g["L"].shape[0] == 12
    assert len(g["priors"]) == g["H"].shape[1] == g["L"].shape[1]
    assert np.isfinite(g["priors"]).all() and (g["priors"] > 0).all()


def test_gross_invalid_distance_rejected():
    with pytest.raises(ValueError, match="distance"):
        make_circuit(Workload("gross", 5, .001, 2, 1, SEED_TEST))
