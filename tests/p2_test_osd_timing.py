# OWNER: P2
"""Day 3-4: paired OSD timing (qdec.costmodel.osd_timing) on P2's own Stim shots."""
import numpy as np
import pytest

from qdec.cache.devshots import sample_dev_shots
from qdec.contracts import SEED_TEST
from qdec.costmodel.osd_timing import REFERENCE_SETTINGS, cpu_ratio, file_sha256, measure_osd


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


def test_payload_records_timed_shots(shots):
    res = measure_osd(shots["H"], shots["priors"], shots["dets"], max_shots=8, repeats=1)
    ids = res["timed_shot_ids"]
    assert len(ids) == len(res["bp_osd_ns_per_shot"]) == res["n_timed"] == 8
    assert len(set(ids)) == 8 and all(shots["dets"][i].any() for i in ids)


def test_cpu_ratio_rescales_to_reference_cpu(shots):
    res = measure_osd(shots["H"], shots["priors"], shots["dets"], max_shots=6, repeats=1)
    ref = np.zeros(len(shots["dets"]))
    ref[res["timed_shot_ids"]] = 1.5 * np.asarray(res["bp_osd_ns_per_shot"])   # a 1.5x slower CPU
    assert cpu_ratio(res, ref) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        cpu_ratio(res, ref[:max(res["timed_shot_ids"])])           # reference misses timed shots


def test_long_calls_are_timed_once(shots, monkeypatch):
    import qdec.costmodel.osd_timing as ot
    monkeypatch.setattr(ot, "LONG_CALL_NS", 0.0)                   # every call counts as long
    res = measure_osd(shots["H"], shots["priors"], shots["dets"], max_shots=3, repeats=3)
    assert res["repeats"] == 1


def test_file_sha256_tracks_content(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"shots")
    b.write_bytes(b"shots")
    assert file_sha256(a) == file_sha256(b)
    b.write_bytes(b"other shots")
    assert file_sha256(a) != file_sha256(b)


def test_budget_stops_after_min_shots(shots):
    res = measure_osd(shots["H"], shots["priors"], shots["dets"], max_shots=500, repeats=1,
                      budget_s=1e-6, min_shots=4)
    assert res["n_timed"] == 4


def test_no_osd_shots_raises(shots):
    with pytest.raises(ValueError, match="OSD"):
        measure_osd(shots["H"], shots["priors"], np.zeros_like(shots["dets"]), max_shots=5)
