# OWNER: P3
"""Produce every figure and summary table under RESULTS_DIR from the service/trace tables.

With --mock, uses qdec.mocks tables in memory and writes PLACEHOLDER figures to results/mock/
(git-ignored) so the pipeline and file layout can be checked before any plot exists.
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
from qdec.results import plots


def placeholder_figures(service_df: pd.DataFrame, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(service_df)
    for fn in plots.ALL_PLOTS:
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
    if args.mock:
        from qdec.mocks import mock_service
        service = pd.concat([mock_service(w.tag, n=w.shots) for w in grid], ignore_index=True)
        placeholder_figures(service, RESULTS_DIR / "mock")
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
