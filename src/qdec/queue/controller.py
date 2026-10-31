# OWNER: P3
"""Backlog-aware controller: cap BP iterations when the decode queue grows (System.FULL)."""
import pandas as pd


class BacklogController:
    """Maps queue depth to a BP iteration budget and re-prices shots under that budget.

    Inputs:  max_iters (budget at empty queue), plus whatever thresholds/policy P3 chooses.
    Owner:   P3
    """

    def __init__(self, max_iters: int, **policy):
        raise NotImplementedError

    def budget(self, queue_depth: int) -> int:
        """Iteration budget for the next window given the current queue depth."""
        raise NotImplementedError

    def reprice(self, row: pd.Series, budget: int, iter_costs: dict) -> float:
        """Service time (ns) of one shot when BP is capped at `budget` iterations.

        Inputs:  row with FUNCTIONAL_COLS fields bp_iters, osd_ran (joined with SERVICE_COLS on
                 shot_id/tag), iter_costs[tag][system] from COSTS_DIR/iter_costs.json.
        Outputs: capped_latency = min(bp_iters, budget) * per_iter_ns
                                  + (osd_ns if bp_iters > budget or osd_ran else 0)
                 (cache-hit shots cost cache_ns). Formula is fixed in contracts.py.
        """
        raise NotImplementedError
