# OWNER: P3
"""Discrete-event simulation of a single decoder fed one decode window every round period.

Window i arrives at a_i = i * round_period_ns and is served FIFO by one decoder:
    start_i = max(a_i, finish_{i-1}),  finish_i = start_i + s_i,  latency_i = finish_i - a_i.
Without a controller the service times are fixed, so the recursion has the closed form
    finish_i = S_i + max_{j<=i} (a_j - S_{j-1}),   S_i = s_0 + ... + s_i
which is evaluated with cumulative sums instead of a Python loop.

Stability: the backlog of a FIFO queue drifts upward without bound iff mean service > period
(utilization > 1), so `stable` is mean(service_ns) <= round_period_ns. A depth-trend test was
tried first and misclassified 10-30% of heavy-tailed runs in both directions. Note that a stable
queue can still build very deep transient backlogs when service times are heavy-tailed.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SimResult:
    """Output of simulate(). Owner: P3."""
    latency_ns: np.ndarray    # float[N] per-window latency = queueing wait + service
    t_ns: np.ndarray          # float[M] sample times for queue_depth
    queue_depth: np.ndarray   # int[M] windows waiting or in service at t_ns
    stable: bool              # False if the backlog grows without bound (utilization >= 1)


def simulate(service_ns: np.ndarray, round_period_ns: float, controller=None) -> SimResult:
    """Feed windows arriving every `round_period_ns` through a FIFO decoder.

    Inputs:  service_ns float[N] = SERVICE_COLS.latency_ns for one (tag, system) from
             COSTS_DIR/service_<tag>.parquet (in shot order);
             controller: optional BacklogController; when given, each window's service time is
             re-priced with controller.reprice(row, controller.budget(depth), iter_costs).
    Outputs: SimResult. queue_depth is sampled at each arrival instant (t_ns[i] = a_i) and counts
             the arriving window plus every earlier window not yet finished.
    Owner:   P3
    """
    if controller is not None:
        raise NotImplementedError("controller-driven simulation is Day 2")
    s = np.asarray(service_ns, dtype=np.float64)
    if s.ndim != 1 or len(s) == 0:
        raise ValueError("service_ns must be a non-empty 1-D array")
    if round_period_ns <= 0 or np.any(s < 0):
        raise ValueError("round_period_ns must be > 0 and service_ns >= 0")

    arrival = np.arange(len(s), dtype=np.float64) * round_period_ns
    S = np.cumsum(s)
    S_prev = S - s
    finish = S + np.maximum.accumulate(arrival - S_prev)
    latency = finish - arrival

    # finish is non-decreasing (FIFO), so windows done by a_i are a prefix of the array
    done = np.searchsorted(finish, arrival, side="right")
    depth = np.arange(1, len(s) + 1) - done
    stable = bool(s.mean() <= round_period_ns)
    return SimResult(latency_ns=latency, t_ns=arrival, queue_depth=depth, stable=stable)


def max_sustainable_rate(
    service_ns: np.ndarray,
    controller=None,
    tol: float = 0.01,
    latency_budget_ns: float | None = None,
    quantile: float = 0.999,
    **sim_kwargs,
) -> float:
    """Highest round rate (rounds per second) at which the decoder keeps up.

    "Keeps up" is simulate(...).stable (utilization <= 1). If latency_budget_ns is given it must
    ALSO hold that the `quantile` of per-window latency (queueing + service) is <= the budget;
    that is the tail-aware definition, since a stable queue can still have huge p99.9 latency.
    Both criteria are monotone in the round period, so the period is found by geometric
    bisection to within relative `tol`, and the rate is reported from the side that passes.

    Inputs:  as simulate() (controller and **sim_kwargs are passed through).
    Outputs: rate in Hz; 0.0 if no round period meets the latency budget.
    Owner:   P3
    """
    s = np.asarray(service_ns, dtype=np.float64)
    if not np.any(s > 0):
        raise ValueError("service_ns must contain a positive service time")

    def ok(period: float) -> bool:
        r = simulate(s, period, controller, **sim_kwargs)
        if not r.stable:
            return False
        return latency_budget_ns is None or np.quantile(r.latency_ns, quantile) <= latency_budget_ns

    # At period >= max service nothing ever queues, so latency == service exactly and no longer
    # period can do better: feasibility is decided there, and the search only goes downward.
    # (Searching upward instead loses all latency precision once arrival times reach ~1e20 ns.)
    hi = float(s.max())
    if not ok(hi):
        return 0.0
    lo = hi / 2
    while ok(lo):
        hi, lo = lo, lo / 2
    while hi / lo > 1 + tol:
        mid = np.sqrt(hi * lo)
        hi, lo = (mid, lo) if ok(mid) else (hi, mid)
    return 1e9 / hi
