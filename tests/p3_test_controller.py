# OWNER: P3
import itertools

import numpy as np
import pandas as pd
import pytest

from qdec.contracts import QUICK_GRID, System
from qdec.mocks import mock_functional, mock_service
from qdec.queue.controller import BacklogController, uncapped_mismatch
from qdec.queue.sim import max_sustainable_rate, simulate

COSTS = {"per_iter_ns": 20.0, "osd_ns": 2000.0, "cache_ns": 10.0}


def test_budget_policy_shape():
    c = BacklogController(max_iters=50, min_iters=10, low_depth=2, high_depth=10)
    assert [c.budget(d) for d in (0, 1, 2)] == [50, 50, 50]
    assert c.budget(6) == 30                       # halfway along the ramp
    assert [c.budget(d) for d in (10, 11, 500)] == [10, 10, 10]
    b = [c.budget(d) for d in range(20)]
    assert all(x >= y for x, y in itertools.pairwise(b))   # never increases with depth


def test_bad_policy_rejected():
    with pytest.raises(ValueError):
        BacklogController(50, min_iters=60)
    with pytest.raises(ValueError):
        BacklogController(50, low_depth=5, high_depth=5)


@pytest.mark.parametrize("path,iters,osd_ran,budget,expect", [
    ("cache", 30, False, 5, 10.0),                 # cache hit: cache_ns, budget irrelevant
    ("bp", 10, False, 50, 10 * 20.0),              # converged under budget: no OSD
    ("bp", 10, False, 5, 5 * 20.0 + 2000.0),       # capped before convergence: OSD
    ("bp", 50, True, 50, 50 * 20.0 + 2000.0),      # never converged: OSD at full budget
    ("bp", 50, True, 10, 10 * 20.0 + 2000.0),      # capping a non-converging shot only helps
    ("bp+osd", 50, True, 10, 10 * 20.0 + 2000.0),
])
def test_reprice_matches_contract_formula(path, iters, osd_ran, budget, expect):
    c = BacklogController(50)
    row = {"path": path, "bp_iters": iters, "osd_ran": osd_ran}
    assert c.reprice(row, budget, COSTS) == pytest.approx(expect)
    assert c.reprice(pd.Series(row), budget, COSTS) == pytest.approx(expect)
    tup = next(pd.DataFrame([row]).itertuples(index=False))
    assert c.reprice(tup, budget, COSTS) == pytest.approx(expect)


def rows_for(tag, n=3000):
    svc = mock_service(tag, n=n)
    full = svc[svc["system"] == System.FULL.value].sort_values("shot_id").reset_index(drop=True)
    func = mock_functional(tag, n=n)
    return full, full[["shot_id", "path"]].merge(func[["shot_id", "bp_iters", "osd_ran"]],
                                                 on="shot_id")


def test_reprice_all_matches_scalar():
    _, rows = rows_for(QUICK_GRID[0].tag, n=500)
    c = BacklogController(50)
    for b in (50, 20, 3):
        vec = c.reprice_all(rows, b, COSTS)
        scalar = [c.reprice(r, b, COSTS) for r in rows.itertuples(index=False)]
        np.testing.assert_allclose(vec, scalar)


def test_controller_that_never_caps_equals_plain_simulation():
    _, rows = rows_for(QUICK_GRID[0].tag)
    c = BacklogController(50, low_depth=10**9, high_depth=10**9 + 1)
    base = c.reprice_all(rows, 50, COSTS)
    with_c = simulate(base, 100.0, c, rows=rows, iter_costs=COSTS)
    plain = simulate(base, 100.0)
    np.testing.assert_allclose(with_c.latency_ns, plain.latency_ns)
    np.testing.assert_array_equal(with_c.queue_depth, plain.queue_depth)
    assert (with_c.budget == 50).all()


def overload_rows(n=20_000, seed=0):
    """BP-only shots with a heavy iteration tail where iterations are expensive vs OSD."""
    rng = np.random.default_rng(seed)
    iters = np.minimum(rng.geometric(0.05, n), 200)
    return pd.DataFrame({"path": "bp", "bp_iters": iters, "osd_ran": iters == 200})


def test_controller_cuts_backlog_when_iterations_dominate():
    rows = overload_rows()
    costs = {"per_iter_ns": 50.0, "osd_ns": 400.0, "cache_ns": 10.0}
    c = BacklogController(200, min_iters=10, low_depth=2, high_depth=6)
    base = c.reprice_all(rows, 200, costs)
    period = 1.02 * base.mean()                    # just barely stable without a controller
    plain = simulate(base, period)
    ctl = simulate(base, period, c, rows=rows, iter_costs=costs)
    assert ctl.queue_depth.max() < plain.queue_depth.max() / 3
    assert np.quantile(ctl.latency_ns, 0.999) < np.quantile(plain.latency_ns, 0.999) / 3
    assert (ctl.budget < 200).any() and ctl.stable
    rate_plain = max_sustainable_rate(base, latency_budget_ns=20 * base.mean())
    rate_ctl = max_sustainable_rate(base, c, latency_budget_ns=20 * base.mean(),
                                    rows=rows, iter_costs=costs)
    assert rate_ctl > rate_plain


def test_capping_costs_more_when_osd_dominates():
    """The documented trade-off: capping a converging shot adds a full OSD call."""
    c = BacklogController(50)
    row = {"path": "bp", "bp_iters": 12, "osd_ran": False}
    assert c.reprice(row, 10, COSTS) > c.reprice(row, 50, COSTS)


def test_simulate_with_controller_needs_rows_and_costs():
    with pytest.raises(ValueError, match="rows"):
        simulate(np.ones(5), 1.0, BacklogController(50))


def test_uncapped_mismatch():
    _, rows = rows_for(QUICK_GRID[0].tag)
    base = BacklogController(50).reprice_all(rows, 50, COSTS)
    assert uncapped_mismatch(base, rows, COSTS, 50) == pytest.approx(0.0)
    assert uncapped_mismatch(base * 2, rows, COSTS, 50) == pytest.approx(0.5)
