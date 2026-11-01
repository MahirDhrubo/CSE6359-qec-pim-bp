# OWNER: P3
"""Produce every figure and summary table under RESULTS_DIR from the service/trace tables.

Always writes results/backlog_explosion_synthetic.png (needs no upstream data).
With --mock, uses qdec.mocks tables in memory and writes to results/mock/ (git-ignored): real
figures for the plots that exist, PLACEHOLDER figures for the rest.
Usage: python scripts/p3_make_results.py --grid quick|full [--mock]
"""
import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from qdec import io
from qdec.contracts import (
    COSTS_DIR,
    FULL_GRID,
    QUICK_GRID,
    RESULTS_DIR,
    SERVICE_COLS,
)
from qdec.queue.synthetic import same_mean_lognormal
from qdec.results import plots

# Day-1 backlog-explosion experiment: same mean (0.9 of the round period), different tails.
SYNTH_PERIOD_NS = 1000.0
SYNTH_SIGMAS = {"light tail (sigma=0.3)": 0.3, "heavy tail (sigma=1.5)": 1.5}
SYNTH_SHOTS, SYNTH_SEED = 20_000, 0
# Round period for mock-data figures: between the mock systems' means, so some keep up and
# bposd_cpu does not.
MOCK_PERIOD_NS = 700.0


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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args(argv)

    grid = QUICK_GRID if args.grid == "quick" else FULL_GRID
    synthetic_backlog_figure()
    if args.mock:
        from qdec.mocks import mock_service
        service = pd.concat([mock_service(w.tag, n=w.shots) for w in grid], ignore_index=True)
        tag = grid[0].tag
        path = plots.plot_backlog_explosion(
            service[service["tag"] == tag], RESULTS_DIR / "mock" / "backlog_explosion.png",
            MOCK_PERIOD_NS, title="Backlog per system", note=f"MOCK DATA, {tag}",
        )
        print(f"wrote {path} (MOCK)")
        placeholder_figures(service, RESULTS_DIR / "mock", skip={plots.plot_backlog_explosion})
        return

    from qdec.results.tables import summary_table

    io.ensure_dirs()
    service = pd.concat(
        [io.read_table(COSTS_DIR / f"service_{w.tag}.parquet", SERVICE_COLS) for w in grid],
        ignore_index=True,
    )
    summary_table(service).to_csv(RESULTS_DIR / "summary.csv", index=False)
    plots.plot_latency_cdf(service, RESULTS_DIR / "latency_cdf.png")
    # P3: call the remaining plot_* functions here as they are implemented.


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"p3_make_results: SKIPPED, missing input ({e}) (run the upstream make target, or MOCK=1)")
        sys.exit(0)
    except NotImplementedError:
        print("p3_make_results: not implemented yet")
        sys.exit(0)
