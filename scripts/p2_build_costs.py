# OWNER: P2
"""Turn functional + cache traces into per-shot service times for every System.

Writes COSTS_DIR/service_<tag>.parquet (SERVICE_COLS) and COSTS_DIR/iter_costs.json.
Reads TRACES_DIR/functional_<tag>.parquet (P1), cache_<tag>.parquet + .json (P2) and, when present,
P1's MEASURED TRACES_DIR/bposd_latency_<tag>.npz for the bposd_cpu rows and P2's
COSTS_DIR/osd_<tag>.json (scripts/p2_measure_osd.py) for the OSD fallback time.
With --mock, writes qdec.mocks.mock_service tables and placeholder iter_costs (MOCK numbers).
Usage: python scripts/p2_build_costs.py --grid quick|full [--mock]
"""
import argparse
import json
import sys

import numpy as np

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


def bposd_latency(tag: str, n: int):
    """P1's measured ldpc BP+OSD wall clock per shot, or None if not handed off (yet)."""
    path = TRACES_DIR / f"bposd_latency_{tag}.npz"
    if not path.exists():
        print(f"  {tag}: no {path.name}; bposd_cpu rows omitted (run scripts/p1_check_baselines.py --all)")
        return None
    z = io.load_npz(path)
    if not np.array_equal(z["shot_id"], np.arange(n)):
        print(f"  {tag}: {path.name} covers {len(z['shot_id'])} shots, not {n}; bposd_cpu omitted")
        return None
    return z["latency_ns"]


def measured_osd_ns(tag: str):
    """Median paired OSD time from scripts/p2_measure_osd.py, or None (yaml fallback)."""
    path = COSTS_DIR / f"osd_{tag}.json"
    if not path.exists():
        print(f"  {tag}: no {path.name}; osd_ns = op_costs.yaml fallback (ASSUMED), "
              f"run scripts/p2_measure_osd.py")
        return None
    return json.loads(path.read_text())["osd_ns"]["p50"]


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

    from qdec.cache.cam import CACHEABLE_CODES
    from qdec.costmodel.service import (
        build_service_table,
        iter_cost_entries,
        load_costs,
        write_iter_costs,
    )

    costs = load_costs()
    entries, missing = {}, []
    for w in grid:
        gname = f"{w.code}_d{w.distance}_p{w.p:g}"
        try:
            stats = json.loads((GRAPHS_DIR / f"{gname}.json").read_text())
            fdf = io.read_table(TRACES_DIR / f"functional_{w.tag}.parquet", FUNCTIONAL_COLS)
            cdf, kw = None, {}
            if w.code in CACHEABLE_CODES:
                cdf = io.read_table(TRACES_DIR / f"cache_{w.tag}.parquet", CACHE_COLS)
                cam = json.loads((TRACES_DIR / f"cache_{w.tag}.json").read_text())
                kw = {"cam_entries": cam["cam_entries"], "key_bits": cam["key_bits"]}
        except FileNotFoundError as e:
            missing.append(f"{w.tag} ({e.filename})")
            continue
        kw["osd_ns"] = measured_osd_ns(w.tag)
        sdf = build_service_table(fdf, cdf, stats, costs,
                                  bposd_latency_ns=bposd_latency(w.tag, len(fdf)), **kw)
        print(f"wrote {io.write_table(sdf, SERVICE_COLS, COSTS_DIR / f'service_{w.tag}.parquet')}"
              f"  ({sdf['system'].nunique()} systems)", flush=True)
        entries[w.tag] = iter_cost_entries(fdf, cdf, stats, costs, **kw)
    if entries:
        print(f"wrote {write_iter_costs(entries, COSTS_DIR / 'iter_costs.json')}  ({len(entries)} tags)")
    if missing:
        print(f"p2_build_costs: SKIPPED {len(missing)} workload(s) with missing inputs "
              f"(run `make data functional cache`): " + "; ".join(missing))


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"p2_build_costs: SKIPPED, missing input ({e}) (run the upstream make target, or MOCK=1)")
        sys.exit(0)
    except NotImplementedError:
        print("p2_build_costs: not implemented yet")
        sys.exit(0)
