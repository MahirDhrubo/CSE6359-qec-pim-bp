# OWNER: P3
"""Every plot function must survive zeros, NaN/Inf and missing systems (real data has all three)."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from qdec.contracts import QUICK_GRID, System
from qdec.mocks import mock_cache, mock_functional, mock_service
from qdec.queue.sim import simulate
from qdec.results import plots
from qdec.results.plots import _log_limits, _plot_rates, failure_rate, wilson_interval

TAG = QUICK_GRID[0].tag


def dirty_service(n=400) -> pd.DataFrame:
    """One tag; zero latencies, NaN/Inf latencies, an all-NaN system, a missing system."""
    df = mock_service(TAG, n=n)
    df = df[df["system"] != System.RELAY_FPGA.value].reset_index(drop=True)  # missing system
    lat = df["latency_ns"].to_numpy().copy()
    pim = (df["system"] == System.PIM_CACHE.value).to_numpy()
    lat[np.flatnonzero(pim)[: n // 2]] = 0.0                   # half the cache shots at 0 ns
    lat[np.flatnonzero(df["system"] == System.BP_CPU_CONV.value)[:5]] = np.nan
    lat[np.flatnonzero(df["system"] == System.FULL.value)[:3]] = np.inf
    lat[(df["system"] == System.NEARMEM_DPU.value).to_numpy()] = np.nan   # nothing left
    return df.assign(latency_ns=lat)


def dirty_functional() -> dict[str, pd.DataFrame]:
    out = {}
    for i, w in enumerate(QUICK_GRID):
        f = mock_functional(w.tag, n=500, seed=i)
        if i % 2 == 0:                                          # zero logical failures
            f["fail_bp_osd"] = False
            f["fail_bp_float"] = False
        out[w.tag] = f                                          # fail_matching is all <NA>
    return out


def dirty_cache() -> dict[str, pd.DataFrame]:
    c = mock_cache(QUICK_GRID[0].tag, n=500)
    c["fail_cache"] = c["fail_cache"].where(c["fail_cache"].isna(), False)   # 0 failures on hits
    return {QUICK_GRID[0].tag: c}                               # other tags have no cache trace


def dirty_rates() -> pd.DataFrame:
    r = pd.DataFrame([{"tag": w.tag, "system": s.value, "rate_hz": 1e6 * (i + 1) / 3}
                      for w in QUICK_GRID for i, s in enumerate(System)])
    r.loc[0, "rate_hz"] = 0.0
    r.loc[1, "rate_hz"] = np.nan
    r.loc[2, "rate_hz"] = np.inf
    return r.drop(index=5)                                      # a missing (tag, system)


def sim_results():
    svc = mock_service(TAG, n=300)
    return {s: simulate(g.sort_values("shot_id")["latency_ns"].to_numpy(), 500.0)
            for s, g in svc.groupby("system")}


# one call per plot function; every function in plots.ALL_PLOTS must appear here
CALLS = {
    "plot_backlog_explosion": lambda out: plots.plot_backlog_explosion(dirty_service(), out, 500.0),
    "plot_ler": lambda out: plots.plot_ler(dirty_functional(), dirty_cache(), out),
    "plot_latency_cdf": lambda out: plots.plot_latency_cdf(dirty_service(), out),
    "plot_backlog_over_time": lambda out: plots.plot_backlog_over_time(sim_results(), out),
    "plot_max_rate": lambda out: plots.plot_max_rate(dirty_rates(), out),
    "plot_hit_rate": lambda out: plots.plot_hit_rate(dirty_cache(), out),
    "plot_bytes_energy": lambda out: plots.plot_bytes_energy(dirty_service(), out),
    "plot_ablation": lambda out: plots.plot_ablation(
        pd.DataFrame({"tag": TAG, "system": ["pim_nocache", "pim_cache", "full"],
                      "p99_ns": [0.0, np.nan, 5.0], "p999_ns": [np.inf, 3.0, 0.0]}), out),
    "plot_iters_hist": lambda out: plots.plot_iters_hist(
        dirty_functional()[TAG].assign(bp_iters=lambda d: d["bp_iters"].where(d.index > 3, 0)),
        out),
}


def test_every_plot_function_is_covered():
    assert {f.__name__ for f in plots.ALL_PLOTS} == set(CALLS)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_plot_survives_zeros_nans_and_missing_systems(name, tmp_path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                     # warnings are expected here
            out = CALLS[name](tmp_path / f"{name}.png")
    except NotImplementedError:
        pytest.skip(f"{name} is still a stub")
    assert out.exists() and out.stat().st_size > 5_000


def test_max_rate_all_zero_regression(tmp_path):
    """The make-results crash: every system misses the latency budget -> NaN axis limit."""
    r = dirty_rates().assign(rate_hz=0.0)
    assert plots.plot_max_rate(r, tmp_path / "r.png").exists()
    assert plots.plot_max_rate(r, tmp_path / "r2.png", target_hz=None).exists()


def test_missing_and_empty_systems_warn(tmp_path):
    with pytest.warns(UserWarning, match="skipping system 'nearmem_dpu'"):
        plots.plot_latency_cdf(dirty_service(), tmp_path / "c.png")
    with pytest.warns(UserWarning, match="dropped 5 NaN/Inf"):
        plots.plot_backlog_explosion(dirty_service(), tmp_path / "b.png", 500.0)
    with pytest.warns(UserWarning, match="no finite rate"):
        plots.plot_max_rate(dirty_rates(), tmp_path / "r.png")
    with pytest.warns(UserWarning, match="skipping 'PyMatching', no data"):
        plots.plot_ler(dirty_functional(), dirty_cache(), tmp_path / "l.png")


def test_rule_of_three():
    assert failure_rate(0, 20_000) == (pytest.approx(1.5e-4), True)
    assert failure_rate(7, 1000) == (pytest.approx(7e-3), False)
    rate, bound = failure_rate(0, 0)
    assert np.isnan(rate) and not bound


def test_wilson_interval_brackets_rate():
    lo, hi = wilson_interval(7, 1000)
    assert lo < 7e-3 < hi
    lo0, hi0 = wilson_interval(0, 1000)
    assert lo0 == 0.0 and 0 < hi0 < 3 / 1000 * 1.5


def test_zero_failure_points_are_drawn_as_upper_bounds_not_dropped():
    fig, ax = plt.subplots()
    drew_bound = _plot_rates(ax, [1e-3, 2e-3, 3e-3], [0, 4, 0], [20_000, 20_000, 10_000],
                             "#2a78d6", "x")
    assert drew_bound
    tri = [ln for ln in ax.lines if ln.get_marker() == "v"]
    assert len(tri) == 1
    np.testing.assert_allclose(tri[0].get_ydata(), [3 / 20_000, 3 / 10_000])
    line = next(ln for ln in ax.lines if ln.get_label() == "x")
    assert len(line.get_xdata()) == 3                          # all three points kept
    plt.close(fig)


@pytest.mark.parametrize("values,expect", [
    ([0.0, np.nan, np.inf, -1.0, 10.0, 100.0], (5.0, 400.0)),
    ([0.0, 0.0], (1.0, 10.0)),                                  # nothing positive -> fallback
    ([], (1.0, 10.0)),
])
def test_log_limits_ignore_non_finite_and_non_positive(values, expect):
    assert _log_limits(values) == pytest.approx(expect)
    lo, hi = _log_limits(values, extra=(np.nan, 1e6))
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi
