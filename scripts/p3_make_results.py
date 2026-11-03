# OWNER: P3
"""Produce every figure and summary table under RESULTS_DIR from the service/trace tables.

Always writes results/backlog_explosion_synthetic.png (needs no upstream data). Then, per run:
  summary.csv                p50 / p99 / p99.9 service latency per (tag, system)
  latency_cdf_<tag>.png      service-latency tail per system
  max_rate.csv / .png        max sustainable round rate per (tag, system); FULL runs the
                             backlog controller inside the queue simulation
With --mock, inputs come from qdec.mocks (plus placeholder iter_costs) and everything goes to
results/mock/ (git-ignored), with PLACEHOLDER figures for plots that don't exist yet.
Without --mock, inputs are data/costs/service_<tag>.parquet, data/costs/iter_costs.json and
data/traces/functional_<tag>.parquet.
Usage: python scripts/p3_make_results.py --grid quick|full [--mock]
           [--latency-budget-ns 10000] [--max-iters N]
"""
import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from qdec import io
from qdec.contracts import (
    CACHE_COLS,
    COSTS_DIR,
    FULL_GRID,
    FUNCTIONAL_COLS,
    QUICK_GRID,
    RESULTS_DIR,
    SERVICE_COLS,
    TRACES_DIR,
    System,
)
from qdec.queue.controller import BacklogController, best_budget, uncapped_mismatch
from qdec.queue.sim import max_sustainable_rate, simulate
from qdec.queue.synthetic import same_mean_lognormal
from qdec.results import plots
from qdec.results.tables import summary_table

# Day-1 backlog-explosion experiment: same mean (0.9 of the round period), different tails.
SYNTH_PERIOD_NS = 1000.0
SYNTH_SIGMAS = {"light tail (sigma=0.3)": 0.3, "heavy tail (sigma=1.5)": 1.5}
SYNTH_SHOTS, SYNTH_SEED = 20_000, 0
# Round period for mock-data backlog figure: between the mock systems' means, so some keep up and
# bposd_cpu does not.
MOCK_PERIOD_NS = 700.0
# Placeholder controller prices for --mock only (same shape as iter_costs.json); NOT a cost model.
MOCK_ITER_COSTS = {"per_iter_ns": 20.0, "osd_ns": 2000.0, "cache_ns": 10.0}
REAL_NOTE = "MODELED (bposd_cpu measured; nearmem_dpu, relay_fpga published)"


def synthetic_backlog_figure() -> None:
    df = same_mean_lognormal(0.9 * SYNTH_PERIOD_NS, SYNTH_SIGMAS, SYNTH_SHOTS, SYNTH_SEED)
    path = plots.plot_backlog_explosion(
        df, RESULTS_DIR / "backlog_explosion_synthetic.png", SYNTH_PERIOD_NS,
        note=f"synthetic lognormal service times, mean 0.9T, seed {SYNTH_SEED}",
    )
    print(f"wrote {path}")


def placeholder_figures(service_df: pd.DataFrame, out_dir, skip=()) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(service_df)
    for fn in plots.ALL_PLOTS:
        if fn in skip:
            continue
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, f"PLACEHOLDER (MOCK DATA)\n{fn.__name__}\n{n} service rows",
                ha="center", va="center")
        ax.set_axis_off()
        path = out_dir / f"{fn.__name__.removeprefix('plot_')}.png"
        fig.savefig(path, dpi=80)
        plt.close(fig)
        print(f"wrote {path} (MOCK placeholder)")


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


def rate_table(service, functional, iter_costs, budget_ns: float, max_iters_arg) -> pd.DataFrame:
    out = []
    for tag, svc in service.groupby("tag", sort=False):
        max_iters = max_iters_arg or int(functional[tag]["bp_iters"].max())
        rows = controller_rows(svc, functional[tag])
        costs = iter_costs.get(tag, {}).get(System.FULL.value)
        if costs is None or rows.empty:
            print(f"warning: {tag}: no iter_costs or FULL rows; full is simulated without "
                  f"the controller")
            for system, g in svc.groupby("system", sort=False):
                s = g.sort_values("shot_id")["latency_ns"].to_numpy()
                out.append({"tag": tag, "system": system,
                            "rate_stable_hz": max_sustainable_rate(s),
                            "rate_hz": max_sustainable_rate(s, latency_budget_ns=budget_ns)})
            continue
        gap = uncapped_mismatch(rows["latency_ns"], rows, costs, max_iters)
        floor = best_budget(rows, costs, max_iters)
        print(f"{tag}: max_iters={max_iters}{'' if max_iters_arg else ' (inferred)'}, "
              f"controller floor b*={floor}, "
              f"service-table vs iter_costs median gap for full = {gap:.0%}")
        ctl = BacklogController(max_iters, min_iters=floor)
        for system, g in svc.groupby("system", sort=False):
            s = g.sort_values("shot_id")["latency_ns"].to_numpy()
            kw = {}
            if system == System.FULL.value:
                # the controller re-prices from iter_costs, so its uncapped base must come from
                # the same model (see the gap printed above)
                s = ctl.reprice_all(rows, max_iters, costs)
                kw = {"controller": ctl, "rows": rows, "iter_costs": costs}
            out.append({
                "tag": tag, "system": system,
                "rate_stable_hz": max_sustainable_rate(s, **kw),
                "rate_hz": max_sustainable_rate(s, latency_budget_ns=budget_ns, **kw),
            })
            if system == System.FULL.value:
                compare_full(tag, svc, s, kw, out[-1]["rate_hz"], out, budget_ns)
    return pd.DataFrame(out)


def compare_full(tag, svc, full_s, kw, full_rate, out, budget_ns) -> None:
    """Day-2 check: does FULL (with controller) beat PIM_CACHE in max rate or p99.9 latency?
    Also prints controller on vs off on the SAME iter_costs pricing, which isolates the
    controller's effect even when the service table and iter_costs disagree."""
    off = max_sustainable_rate(full_s, latency_budget_ns=budget_ns)
    print(f"  full, controller on vs off (same pricing): max rate {full_rate / 1e6:.2f} vs "
          f"{off / 1e6:.2f} MHz")
    pim = next((r for r in out if r["tag"] == tag and r["system"] == System.PIM_CACHE.value), None)
    if pim is None or not pim["rate_hz"] > 0:
        return
    period = 1e9 / pim["rate_hz"]
    s_pim = svc[svc["system"] == System.PIM_CACHE.value].sort_values("shot_id")["latency_ns"]
    p_pim = pd.Series(simulate(s_pim.to_numpy(), period).latency_ns).quantile(0.999)
    p_full = pd.Series(simulate(full_s, period, **kw).latency_ns).quantile(0.999)
    better = full_rate > pim["rate_hz"] or p_full < p_pim
    print(f"  full vs pim_cache @ {period:,.0f} ns rounds: max rate {full_rate / 1e6:.2f} vs "
          f"{pim['rate_hz'] / 1e6:.2f} MHz, p99.9 {p_full:,.0f} vs {p_pim:,.0f} ns "
          f"-> full {'BETTER' if better else 'NOT better'} (budget {budget_ns:,.0f} ns)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--latency-budget-ns", type=float, default=10_000.0,
                    help="p99.9 latency (queueing + service) a sustainable rate must meet")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="BP iteration cap; inferred as max(bp_iters) per tag if omitted")
    args = ap.parse_args(argv)

    grid = QUICK_GRID if args.grid == "quick" else FULL_GRID
    io.ensure_dirs()
    synthetic_backlog_figure()
    service, functional, cache, iter_costs = load_inputs(grid, args.mock)
    out_dir = RESULTS_DIR / "mock" if args.mock else RESULTS_DIR
    note = "MOCK DATA" if args.mock else REAL_NOTE
    out_dir.mkdir(parents=True, exist_ok=True)

    ler_note = "MOCK DATA" if args.mock else f"{len(functional)} workloads, SEED_TEST shots"
    print(f"wrote {plots.plot_ler(functional, cache, out_dir / 'ler.png', note=ler_note)}")
    if cache:
        print(f"wrote {plots.plot_hit_rate(cache, out_dir / 'hit_rate.png', note=ler_note)}")

    summary_table(service).to_csv(out_dir / "summary.csv", index=False)
    print(f"wrote {out_dir / 'summary.csv'}")
    for tag, svc in service.groupby("tag", sort=False):
        print(f"wrote {plots.plot_latency_cdf(svc, out_dir / f'latency_cdf_{tag}.png', note=note)}")

    rates = rate_table(service, functional, iter_costs, args.latency_budget_ns, args.max_iters)
    rates.to_csv(out_dir / "max_rate.csv", index=False)
    crit = f"p99.9 latency <= {args.latency_budget_ns / 1e3:g} us, full uses the controller"
    print(f"wrote {plots.plot_max_rate(rates, out_dir / 'max_rate.png', note=f'{crit}  |  {note}')}")

    if args.mock:
        tag = grid[0].tag
        path = plots.plot_backlog_explosion(
            service[service["tag"] == tag], out_dir / "backlog_explosion.png",
            MOCK_PERIOD_NS, title="Backlog per system", note=f"MOCK DATA, {tag}",
        )
        print(f"wrote {path} (MOCK)")
        done = {plots.plot_backlog_explosion, plots.plot_latency_cdf, plots.plot_max_rate,
                plots.plot_ler, plots.plot_hit_rate}
        placeholder_figures(service, out_dir, skip=done)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"p3_make_results: SKIPPED, missing input ({e}) (run the upstream make target, or MOCK=1)")
        sys.exit(0)
    except NotImplementedError:
        print("p3_make_results: not implemented yet")
        sys.exit(0)
