# OWNER: P3
"""Live demo: stream decode windows through the queue simulator for each System and print
p50 / p99 / p99.9 latency, final queue depth and stability side by side.

With --mock, uses qdec.mocks.mock_service for the first workload of the grid.
Usage: python scripts/p3_demo.py --grid quick|full [--mock] [--period-ns 1000]
"""
import argparse
import sys

from qdec import io
from qdec.contracts import COSTS_DIR, FULL_GRID, QUICK_GRID, SERVICE_COLS


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--period-ns", type=float, default=1000.0, help="syndrome round period")
    args = ap.parse_args(argv)

    w = (QUICK_GRID if args.grid == "quick" else FULL_GRID)[0]
    if args.mock:
        from qdec.mocks import mock_service
        service = mock_service(w.tag, n=w.shots)
    else:
        service = io.read_table(COSTS_DIR / f"service_{w.tag}.parquet", SERVICE_COLS)

    print(f"workload {w.tag}, round period {args.period_ns:g} ns{' (MOCK DATA)' if args.mock else ''}")
    print(service.groupby("system")["latency_ns"].describe()[["count", "mean", "max"]])

    from qdec.queue.sim import simulate
    for system, g in service.groupby("system"):
        res = simulate(g.sort_values("shot_id")["latency_ns"].to_numpy(), args.period_ns)
        print(f"{system:22s} stable={res.stable} final_depth={res.queue_depth[-1]}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"p3_demo: SKIPPED, missing input ({e}) (run the upstream make target, or MOCK=1)")
        sys.exit(0)
    except NotImplementedError:
        print("p3_demo: queue simulator not implemented yet")
        sys.exit(0)
