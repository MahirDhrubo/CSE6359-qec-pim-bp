# OWNER: P2
"""Day 3-4: paired OSD timing (qdec.costmodel.osd_timing) on P2's own Stim shots."""
import numpy as np
import pytest

from qdec.cache.devshots import sample_dev_shots
from qdec.contracts import SEED_TEST
from qdec.costmodel.osd_timing import REFERENCE_SETTINGS, measure_osd


@pytest.fixture(scope="module")
def shots():
    return sample_dev_shots(5, 3e-3, 600, SEED_TEST)   # p = 3e-3: plenty of OSD shots


def test_measure_osd_payload(shots):
    res = measure_osd(shots["H"], shots["priors"], shots["dets"], max_shots=10, repeats=1)
    assert res["n_timed"] == 10 and res["converge_mismatch"] == 0
    assert res["osd_ns"]["p50"] > 0 and res["bp_osd_ns"]["p50"] > res["bp_only_ns"]["p50"]
    assert res["settings"] == REFERENCE_SETTINGS
    assert res["n_test_shots"] == len(shots["dets"])
    assert 0 < res["n_osd_shots_est"] < len(shots["dets"])


def test_budget_stops_after_min_shots(shots):
    res = measure_osd(shots["H"], shots["priors"], shots["dets"], max_shots=500, repeats=1,
                      budget_s=1e-6, min_shots=4)
    assert res["n_timed"] == 4


def test_no_osd_shots_raises(shots):
    with pytest.raises(ValueError, match="OSD"):
        measure_osd(shots["H"], shots["priors"], np.zeros_like(shots["dets"]), max_shots=5)
