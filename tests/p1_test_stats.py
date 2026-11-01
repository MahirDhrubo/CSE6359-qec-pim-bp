# OWNER: P1
import numpy as np
import pytest

from qdec.workload.stats import cis_overlap, latency_percentiles, wilson_ci


def test_wilson_contains_estimate_and_stays_in_bounds():
    for k, n in [(0, 100), (4, 20_000), (50, 100), (100, 100)]:
        lo, hi = wilson_ci(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_wilson_zero_failures_is_not_degenerate():
    n, z = 20_000, 1.96
    lo, hi = wilson_ci(0, n, z)
    assert lo == 0.0 and hi == pytest.approx(z * z / (n + z * z))  # Wilson upper bound at k=0


def test_wilson_known_value():
    lo, hi = wilson_ci(10, 100)
    assert lo == pytest.approx(0.0552, abs=1e-4) and hi == pytest.approx(0.1744, abs=1e-4)


def test_wilson_narrows_with_more_shots():
    a, b = wilson_ci(10, 1_000), wilson_ci(100, 10_000)
    assert (b[1] - b[0]) < (a[1] - a[0])


def test_cis_overlap():
    assert cis_overlap((0.1, 0.2), (0.15, 0.3)) and cis_overlap((0.1, 0.2), (0.2, 0.3))
    assert not cis_overlap((0.1, 0.2), (0.21, 0.3))


def test_latency_percentiles_in_us():
    r = latency_percentiles(np.arange(1, 1001) * 1e3)  # 1..1000 us
    assert r["p50"] == pytest.approx(500.5) and r["max"] == 1000.0
    assert list(r) == ["p50", "p90", "p99", "p99.9", "max"]
