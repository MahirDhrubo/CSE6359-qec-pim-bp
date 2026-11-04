# OWNER: P3
"""Input loading and per-system simulation setup shared by scripts/p3_make_results.py and
scripts/p3_demo.py, so both always simulate exactly the same thing."""
import json

import pandas as pd

from qdec import io
from qdec.contracts import (
    CACHE_COLS,
    COSTS_DIR,
    FUNCTIONAL_COLS,
    SERVICE_COLS,
    TRACES_DIR,
    System,
)
from qdec.queue.controller import BacklogController, best_budget, uncapped_mismatch

# Placeholder controller prices for --mock only (same shape as iter_costs.json); NOT a cost model.
MOCK_ITER_COSTS = {"per_iter_ns": 20.0, "osd_ns": 2000.0, "cache_ns": 10.0}


def load_inputs(grid, mock: bool):
    """-> service DataFrame (all tags), {tag: functional df}, {tag: cache df},
    {tag: {system: costs}}. A workload whose service table or functional trace is missing is
    skipped with a warning (FULL_GRID lands gradually); FileNotFoundError only if none exist.
    A missing cache trace is left out (plot_ler / plot_hit_rate skip that series)."""
    if mock:
        from qdec.mocks import mock_cache, mock_functional, mock_service
        service = pd.concat([mock_service(w.tag, n=w.shots) for w in grid], ignore_index=True)
        functional = {w.tag: mock_functional(w.tag, n=w.shots) for w in grid}
        cache = {w.tag: mock_cache(w.tag, n=w.shots) for w in grid}
        iter_costs = {w.tag: {s.value: dict(MOCK_ITER_COSTS) for s in System} for w in grid}
        return service, functional, cache, iter_costs
    services, functional, cache = [], {}, {}
    for w in grid:
        s_path = COSTS_DIR / f"service_{w.tag}.parquet"
        f_path = TRACES_DIR / f"functional_{w.tag}.parquet"
        missing = [str(p) for p in (s_path, f_path) if not p.exists()]
        if missing:
            print(f"warning: skipping {w.tag}, missing {', '.join(missing)}")
            continue
        services.append(io.read_table(s_path, SERVICE_COLS))
        functional[w.tag] = io.read_table(f_path, FUNCTIONAL_COLS)
    if not services:
        raise FileNotFoundError(f"no service table for any workload in {COSTS_DIR}")
    service = pd.concat(services, ignore_index=True)
    for w in grid:
        if w.tag not in functional:
            continue
        path = TRACES_DIR / f"cache_{w.tag}.parquet"
        if path.exists():
            cache[w.tag] = io.read_table(path, CACHE_COLS)
        else:
            print(f"warning: {path} missing; LER figure has no cache series for {w.tag}")
    iter_costs = json.loads((COSTS_DIR / "iter_costs.json").read_text())
    return service, functional, cache, iter_costs


def controller_rows(service_tag: pd.DataFrame, functional_tag: pd.DataFrame) -> pd.DataFrame:
    """FULL's service rows in shot order joined with bp_iters / osd_ran from the functional trace."""
    full = service_tag[service_tag["system"] == System.FULL.value].sort_values("shot_id")
    rows = full[["shot_id", "path", "latency_ns"]].merge(
        functional_tag[["shot_id", "bp_iters", "osd_ran"]], on="shot_id", how="left",
        validate="one_to_one")
    if rows["bp_iters"].isna().any():
        raise ValueError("functional trace is missing shots that appear in the service table")
    return rows.reset_index(drop=True)


def system_inputs(tag, svc, functional_tag, iter_costs, max_iters_arg, verbose=False) -> dict:
    """{system: (service_ns, simulate kwargs)} for one workload. FULL runs the backlog
    controller when iter_costs has the tag; its service base is then re-priced from iter_costs
    (uncapped), so controller on/off compare on one cost model (see the printed gap)."""
    out = {system: (g.sort_values("shot_id")["latency_ns"].to_numpy(), {})
           for system, g in svc.groupby("system", sort=False)}
    rows = controller_rows(svc, functional_tag)
    costs = iter_costs.get(tag, {}).get(System.FULL.value)
    if costs is None or rows.empty:
        if verbose:
            print(f"warning: {tag}: no iter_costs or FULL rows; full is simulated without "
                  f"the controller")
        return out
    max_iters = max_iters_arg or int(functional_tag["bp_iters"].max())
    floor = best_budget(rows, costs, max_iters)
    if verbose:
        gap = uncapped_mismatch(rows["latency_ns"], rows, costs, max_iters)
        print(f"{tag}: max_iters={max_iters}{'' if max_iters_arg else ' (inferred)'}, "
              f"controller floor b*={floor}, "
              f"service-table vs iter_costs cost-weighted gap for full = {gap:.0%}")
    ctl = BacklogController(max_iters, min_iters=floor)
    out[System.FULL.value] = (ctl.reprice_all(rows, max_iters, costs),
                              {"controller": ctl, "rows": rows, "iter_costs": costs})
    return out


def reference_period(inputs: dict, utilization: float = 0.9) -> float:
    """Round period at which PIM_CACHE (else the fastest system) runs at `utilization`."""
    ref = inputs.get(System.PIM_CACHE.value) or min(inputs.values(), key=lambda v: v[0].mean())
    return float(ref[0].mean() / utilization)
