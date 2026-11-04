# OWNER: P2
"""Turn functional + cache traces into per-shot service times for every System.

Writes COSTS_DIR/service_<tag>.parquet (SERVICE_COLS) and COSTS_DIR/iter_costs.json.
Reads TRACES_DIR/functional_<tag>.parquet (P1), cache_<tag>.parquet + .json (P2) and, when present,
P1's MEASURED TRACES_DIR/bposd_latency_<tag>.npz for the bposd_cpu rows and P2's
COSTS_DIR/osd_<tag>.json for the OSD fallback time, measuring it first (--osd-budget-s per
tag) when missing or taken on other shots, so `make costs` from a fresh clone never silently
uses the yaml fallback.
The OSD time is measured on this CPU, then rescaled to the CPU that timed bposd_cpu: x the median
ratio of P1's BP+OSD time to ours on the same timed shots (osd_timing.cpu_ratio). A tag without
P1 latencies borrows the median ratio of the other tags of its code; the ratio used is written
back into osd_<tag>.json ("cpu_calibration").
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
    SHOTS_DIR,
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


def osd_measurement(w, budget_s: float):
    """This workload's paired OSD payload (COSTS_DIR/osd_<tag>.json), or None.

    A missing file, or one taken on other shots (QUICK, FULL and reduced profiles share tags, and
    machines sample different shots from the same seed), is measured now within budget_s;
    budget_s <= 0 or a failed measurement means the op_costs.yaml fallback (ASSUMED)."""
    from qdec.costmodel.osd_timing import file_sha256, summary, write_osd

    path = COSTS_DIR / f"osd_{w.tag}.json"
    if path.exists():
        res = json.loads(path.read_text())
        if ("timed_shot_ids" in res
                and res.get("shots_sha256") == file_sha256(SHOTS_DIR / f"{w.tag}.npz")):
            return res
    if budget_s <= 0:
        print(f"  {w.tag}: no current {path.name}; osd_ns = op_costs.yaml fallback (ASSUMED)")
        return None
    try:
        _, res = write_osd(w, budget_s=budget_s)
    except (FileNotFoundError, ValueError) as e:
        print(f"  {w.tag}: OSD not measured ({e}); osd_ns = op_costs.yaml fallback (ASSUMED)")
        return None
    print(f"  measured {summary(res)}", flush=True)
    return res


def calibrated_osd_ns(work: dict) -> dict:
    """{tag: osd_ns or None}: each measured median OSD time expressed on P1's bposd_cpu CPU.

    work[tag] = (workload, osd payload or None, P1 latency per shot or None). The ratio and its
    source are recorded in osd_<tag>.json, so the service tables can be traced back to it."""
    from qdec.costmodel.osd_timing import cpu_ratio

    own = {t: cpu_ratio(res, lat) for t, (_, res, lat) in work.items()
           if res is not None and lat is not None}
    out = {}
    for tag, (w, res, _) in work.items():
        if res is None:
            out[tag] = None
            continue
        same = sorted(own[t] for t, (v, _, _) in work.items() if t in own and v.code == w.code)
        if tag in own:
            ratio, how = own[tag], f"P1 bposd_latency_{tag}.npz on the {res['n_timed']} timed shots"
        elif same:
            ratio = float(np.median(same))
            how = f"median of {len(same)} other {w.code} tags (no P1 latency for this tag)"
        else:
            ratio, how = 1.0, "none: no P1 latency for this code, this CPU's time used as is"
        raw = res["osd_ns"]["p50"]
        res["cpu_calibration"] = {"ratio": ratio, "source": how, "osd_ns_p50_raw": raw,
                                  "osd_ns_p50_used": raw * ratio}
        (COSTS_DIR / f"osd_{tag}.json").write_text(json.dumps(res, indent=2))
        print(f"  {tag}: OSD median {raw / 1e3:,.1f} us here x {ratio:.3f} = "
              f"{raw * ratio / 1e3:,.1f} us on P1's CPU ({how})", flush=True)
        out[tag] = raw * ratio
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--osd-budget-s", type=float, default=60.0,
                    help="seconds to measure OSD for a tag without osd_<tag>.json (0 = use fallback)")
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
    inputs, missing = {}, []
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
        inputs[w.tag] = (w, stats, fdf, cdf, kw, bposd_latency(w.tag, len(fdf)))

    # every OSD measurement first: a tag without P1 latencies borrows its code's CPU ratio
    osd_ns = calibrated_osd_ns({t: (w, osd_measurement(w, args.osd_budget_s), lat)
                                for t, (w, _, _, _, _, lat) in inputs.items()})
    entries = {}
    for tag, (_w, stats, fdf, cdf, kw, lat) in inputs.items():
        kw["osd_ns"] = osd_ns[tag]
        sdf = build_service_table(fdf, cdf, stats, costs, bposd_latency_ns=lat, **kw)
        print(f"wrote {io.write_table(sdf, SERVICE_COLS, COSTS_DIR / f'service_{tag}.parquet')}"
              f"  ({sdf['system'].nunique()} systems)", flush=True)
        entries[tag] = iter_cost_entries(fdf, cdf, stats, costs, **kw)
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
