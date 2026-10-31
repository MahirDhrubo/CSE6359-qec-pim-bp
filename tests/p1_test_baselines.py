# OWNER: P1
import dataclasses
import json

import numpy as np
import pytest
import scipy.sparse as sp

from qdec.contracts import QUICK_GRID
from qdec.workload.baselines import (
    MAX_ITERS,
    OSD_ORDER,
    bposd_decode_batch,
    measurement_info,
)
from qdec.workload.circuits import make_circuit, sample_shots
from qdec.workload.graphs import dem_to_graph

KEYS = {"fail", "bp_iters", "bp_converged", "osd_ran", "latency_ns"}


def _repetition_code(n=5):
    H = sp.csr_matrix(np.eye(n - 1, n, dtype=np.uint8) + np.eye(n - 1, n, k=1, dtype=np.uint8))
    L = sp.csr_matrix(np.eye(1, n, dtype=np.uint8))  # logical = bit 0
    return H, L, np.full(n, 0.05)


def test_repetition_code_single_errors_corrected():
    H, L, priors = _repetition_code()
    errs = np.eye(5, dtype=np.uint8)
    dets = (H @ errs.T % 2).T.astype(bool)
    obs = (L @ errs.T % 2).T.astype(bool)
    r = bposd_decode_batch(H, L, priors, dets, obs, max_iters=20, osd_order=0)
    assert set(r) == KEYS
    assert not r["fail"].any() and r["bp_converged"].all()


def test_wrong_correction_counted_as_failure():
    # flip the true observable: a correct decode now looks like a logical failure
    H, L, priors = _repetition_code()
    dets = np.zeros((3, 4), dtype=bool)
    obs = np.array([[False], [True], [False]])
    r = bposd_decode_batch(H, L, priors, dets, obs, max_iters=20, osd_order=0)
    assert r["fail"].tolist() == [False, True, False]


@pytest.fixture(scope="module")
def surface():
    w = dataclasses.replace(QUICK_GRID[1], shots=400)  # d=5, p=3e-3: plenty of non-converged BP
    g = dem_to_graph(make_circuit(w))
    s = sample_shots(w)
    return g, s, bposd_decode_batch(g["H"], g["L"], g["priors"], s["dets"], s["obs"],
                                    MAX_ITERS, OSD_ORDER)


def test_surface_output_shapes_and_dtypes(surface):
    _, s, r = surface
    S = s["dets"].shape[0]
    assert set(r) == KEYS and all(len(v) == S for v in r.values())
    assert r["fail"].dtype == bool and r["bp_converged"].dtype == bool
    assert r["osd_ran"].dtype == bool and r["latency_ns"].dtype == np.float64
    assert np.issubdtype(r["bp_iters"].dtype, np.integer)


def test_iteration_contract(surface):
    _, s, r = surface
    assert (r["bp_iters"][~r["bp_converged"]] == MAX_ITERS).all()
    assert (r["bp_iters"] >= 0).all() and (r["bp_iters"] <= MAX_ITERS).all()
    assert (r["osd_ran"] == ~r["bp_converged"]).all()
    empty = ~s["dets"].any(axis=1)
    assert empty.any() and (r["bp_iters"][empty] == 0).all() and r["bp_converged"][empty].all()
    assert (~r["bp_converged"]).any()  # this workload must exercise the OSD path


def test_latency_measured_per_shot(surface):
    _, _, r = surface
    lat = r["latency_ns"]
    assert (lat > 0).all() and len(np.unique(lat)) > 1
    # OSD shots should be slower than converged BP shots in the tail
    assert np.median(lat[r["osd_ran"]]) > np.median(lat[~r["osd_ran"]])


def test_ler_reasonable(surface):
    _, _, r = surface
    assert r["fail"].mean() < 0.05  # raw observable flip rate at this point is ~15%


def test_measurement_info_is_json_and_labeled():
    info = measurement_info(max_iters=MAX_ITERS, osd_order=OSD_ORDER)
    assert info["label"] == "MEASURED" and info["threads"] == 1 and info["cpu"]
    assert info["max_iters"] == MAX_ITERS
    json.dumps(info)
