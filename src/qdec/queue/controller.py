# OWNER: P3
"""Backlog-aware controller: cap BP iterations when the decode queue grows (System.FULL).

Pricing follows the formula fixed in contracts.py, per shot, given iter_costs[tag][system]:
    cache path : cache_ns
    otherwise  : min(bp_iters, budget) * per_iter_ns + (osd_ns if bp_iters > budget or osd_ran else 0)
Capping trades BP iterations for an OSD call, so it only shortens a shot whose remaining
iterations cost more than osd_ns; that trade-off is what the policy thresholds tune.
"""
import numpy as np
import pandas as pd

ROW_COLS = ("path", "bp_iters", "osd_ran")   # SERVICE_COLS.path + FUNCTIONAL_COLS iteration fields


class BacklogController:
    """Maps queue depth to a BP iteration budget and re-prices shots under that budget.

    Policy: full budget while queue_depth <= low_depth, min_iters at queue_depth >= high_depth,
    linear in between (rounded). Defaults are provisional; the Day-3 ablation sweeps them.
    Owner:   P3
    """

    def __init__(self, max_iters: int, min_iters: int | None = None,
                 low_depth: int = 2, high_depth: int = 10):
        if min_iters is None:
            min_iters = max(1, max_iters // 5)
        if not 1 <= min_iters <= max_iters or not 0 <= low_depth < high_depth:
            raise ValueError("need 1 <= min_iters <= max_iters and 0 <= low_depth < high_depth")
        self.max_iters, self.min_iters = max_iters, min_iters
        self.low_depth, self.high_depth = low_depth, high_depth

    def budget(self, queue_depth: int) -> int:
        """Iteration budget for the next window given the current queue depth."""
        if queue_depth <= self.low_depth:
            return self.max_iters
        if queue_depth >= self.high_depth:
            return self.min_iters
        frac = (queue_depth - self.low_depth) / (self.high_depth - self.low_depth)
        return round(self.max_iters - frac * (self.max_iters - self.min_iters))

    def reprice(self, row, budget: int, iter_costs: dict) -> float:
        """Service time (ns) of one shot when BP is capped at `budget` iterations.

        Inputs:  row with attributes/keys path (SERVICE_COLS), bp_iters, osd_ran
                 (FUNCTIONAL_COLS, joined on shot_id); iter_costs = iter_costs[tag][system] from
                 COSTS_DIR/iter_costs.json, i.e. {"per_iter_ns", "osd_ns", "cache_ns"}.
        Outputs: capped latency in ns (formula in the module docstring).
        """
        get = row.get if hasattr(row, "get") else lambda k: getattr(row, k)  # Series/dict/tuple
        if get("path") == "cache":
            return float(iter_costs["cache_ns"])
        iters = int(get("bp_iters"))
        osd = iters > budget or bool(get("osd_ran"))
        return min(iters, budget) * iter_costs["per_iter_ns"] + (iter_costs["osd_ns"] if osd else 0.0)

    def reprice_all(self, rows: pd.DataFrame, budget: int, iter_costs: dict) -> np.ndarray:
        """Vectorized reprice() over every row (same formula), float[len(rows)]."""
        iters = rows["bp_iters"].to_numpy()
        osd = (iters > budget) | rows["osd_ran"].to_numpy(dtype=bool)
        bp = np.minimum(iters, budget) * iter_costs["per_iter_ns"] + np.where(
            osd, iter_costs["osd_ns"], 0.0)
        return np.where(rows["path"].to_numpy() == "cache", float(iter_costs["cache_ns"]), bp)


def best_budget(rows: pd.DataFrame, iter_costs: dict, max_iters: int) -> int:
    """Budget b* in [1, max_iters] that minimizes MEAN service over these shots.

    Use as the controller's min_iters: a backlog drains at the mean service rate, so capping
    below b* only adds OSD calls and slows draining. b* == max_iters means capping never pays
    (e.g. osd_ns large vs per_iter_ns), and the controller then never caps. Ties go to the
    larger budget (fewer OSD calls, closer to uncapped decoding).
    Calibrated on the bp_iters distribution, like choosing max_iters itself.
    """
    c = BacklogController(max_iters)
    means = {b: c.reprice_all(rows, b, iter_costs).mean() for b in range(max_iters, 0, -1)}
    return min(means, key=means.get)          # dict order: max_iters first, so ties keep larger b


def uncapped_mismatch(service_ns: np.ndarray, rows: pd.DataFrame, iter_costs: dict,
                      max_iters: int) -> float:
    """Cost-weighted gap between P2's service latency and reprice() at the full budget:
    sum |reprice - service| / sum service.

    Near 0 means service_<tag>.parquet and iter_costs.json describe the same cost model, so the
    controller only changes what capping changes. Large values mean they disagree (expected for
    mocks); report it rather than silently comparing two different models. (A median relative
    error was used first; it read 64% on real data whose means agree to 0.01%, because ~2 ns
    vs 0 ns cache hits dominate a per-shot relative error.)
    """
    full = BacklogController(max_iters).reprice_all(rows, max_iters, iter_costs)
    s = np.asarray(service_ns, dtype=np.float64)
    total = s.sum()
    return float(np.abs(full - s).sum() / total) if total > 0 else float("nan")
