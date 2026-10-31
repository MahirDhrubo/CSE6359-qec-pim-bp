# OWNER: P2
"""Turn functional + cache traces into per-shot service times for every System.

Writes COSTS_DIR/service_<tag>.parquet (SERVICE_COLS) and COSTS_DIR/iter_costs.json.
With --mock, writes qdec.mocks.mock_service tables and placeholder iter_costs (MOCK numbers).
Usage: python scripts/p2_build_costs.py --grid quick|full [--mock]
"""
import argparse
import json
import sys

from qdec import io
from qdec.contracts import (
    CACHE_COLS,
    COSTS_DIR,
    FULL_GRID,
    FUNCTIONAL_COLS,
    GRAPHS_DIR,
    QUICK_GRID,
    SERVICE_COLS,
    TRACES_DIR,
    System,
)

# Placeholder prices for --mock only; they are NOT a cost model.
_MOCK_ITER = {"per_iter_ns": 20.0, "osd_ns": 2000.0, "cache_ns": 10.0}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args(argv)

    io.ensure_dirs()
    grid = QUICK_GRID if args.grid == "quick" else FULL_GRID

    if args.mock:
        from qdec.mocks import mock_service
        for w in grid:
            path = io.write_table(mock_service(w.tag, n=w.shots), SERVICE_COLS,
                                  COSTS_DIR / f"service_{w.tag}.parquet")
            print(f"wrote {path} (MOCK)")
        entries = {w.tag: {s.value: dict(_MOCK_ITER) for s in System} for w in grid}
        path = COSTS_DIR / "iter_costs.json"
        path.write_text(json.dumps(entries, indent=1))
        print(f"wrote {path} (MOCK)")
        return

    from qdec.costmodel.service import build_service_table, load_costs, write_iter_costs

    costs = load_costs()
    for w in grid:
        gname = f"{w.code}_d{w.distance}_p{w.p:g}"
        stats = json.loads((GRAPHS_DIR / f"{gname}.json").read_text())
        fdf = io.read_table(TRACES_DIR / f"functional_{w.tag}.parquet", FUNCTIONAL_COLS)
        cdf = io.read_table(TRACES_DIR / f"cache_{w.tag}.parquet", CACHE_COLS)
        sdf = build_service_table(fdf, cdf, stats, costs)
        print(f"wrote {io.write_table(sdf, SERVICE_COLS, COSTS_DIR / f'service_{w.tag}.parquet')}")
    write_iter_costs({}, COSTS_DIR / "iter_costs.json")  # P2: fill entries per tag/system


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"p2_build_costs: SKIPPED, missing input ({e}) (run the upstream make target, or MOCK=1)")
        sys.exit(0)
    except NotImplementedError:
        print("p2_build_costs: not implemented yet")
        sys.exit(0)
