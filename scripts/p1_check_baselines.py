# OWNER: P1
"""Day-1 sanity check: BP+OSD vs PyMatching LER (95% Wilson CIs) and per-shot latency.

Reads SHOTS_DIR/<tag>.npz (SEED_TEST) and GRAPHS_DIR/<graph>.npz written by `make data`, and
writes the measured BP+OSD latency hand-off for P2:
    TRACES_DIR/bposd_latency_<tag>.npz   shot_id int64[S], latency_ns float64[S]  (MEASURED)
    TRACES_DIR/bposd_latency_<tag>.json  baselines.measurement_info() provenance
Usage: python scripts/p1_check_baselines.py [--grid quick|full] [--all] [--shots N]
Default: surface d=5, p=1e-3 only (the Day-1 "done when" check).
"""
import argparse
import json
import sys

import numpy as np

from qdec import io
from qdec.contracts import FULL_GRID, GRAPHS_DIR, QUICK_GRID, SHOTS_DIR, TRACES_DIR
from qdec.workload.baselines import (
    ALPHA,
    BP_METHOD,
    MAX_ITERS,
    OSD_METHOD,
    OSD_ORDER,
    bposd_decode_batch,
    matching_decode_batch,
    measurement_info,
)
from qdec.workload.circuits import make_circuit
from qdec.workload.selection import select_workloads
from qdec.workload.stats import cis_overlap, latency_percentiles, wilson_ci


def graph_name(w) -> str:
    return f"{w.code}_d{w.distance}_p{w.p:g}"


def check(w, shots: int | None = None) -> dict:
    s = io.load_npz(SHOTS_DIR / f"{w.tag}.npz")
    g = io.load_npz(GRAPHS_DIR / f"{graph_name(w)}.npz")
    dets, obs = s["dets"][:shots], s["obs"][:shots]
    n = dets.shape[0]

    bp = bposd_decode_batch(g["H"], g["L"], g["priors"], dets, obs, MAX_ITERS, OSD_ORDER)
    settings = {"max_iters": MAX_ITERS, "osd_order": OSD_ORDER, "bp_method": BP_METHOD,
                "osd_method": OSD_METHOD, "alpha": ALPHA, "tag": w.tag, "n_shots": n}
    io.save_npz(TRACES_DIR / f"bposd_latency_{w.tag}.npz",
                shot_id=np.arange(n, dtype=np.int64), latency_ns=bp["latency_ns"])
    (TRACES_DIR / f"bposd_latency_{w.tag}.json").write_text(
        json.dumps(measurement_info(**settings), indent=2))

    out = {"tag": w.tag, "n": n, "bposd": _summary(bp, n)}
    out["bposd"]["osd_frac"] = float(bp["osd_ran"].mean())
    if w.code == "surface":
        out["matching"] = _summary(matching_decode_batch(make_circuit(w), dets, obs), n)
        out["cis_overlap"] = cis_overlap(out["bposd"]["ci"], out["matching"]["ci"])
    return out


def _summary(r: dict, n: int) -> dict:
    k = int(r["fail"].sum())
    return {"fails": k, "ler": k / n, "ci": wilson_ci(k, n), "lat_us": latency_percentiles(r["latency_ns"])}


def _print(res: dict) -> None:
    print(f"\n{res['tag']}  ({res['n']} TEST shots)")
    for name in ("bposd", "matching"):
        if name not in res:
            continue
        r = res[name]
        lat = "  ".join(f"{k}={v:,.1f}" for k, v in r["lat_us"].items())
        extra = f"  osd_ran={r['osd_frac']:.1%}" if "osd_frac" in r else ""
        print(f"  {name:9s} LER={r['ler']:.2e} [{r['ci'][0]:.2e}, {r['ci'][1]:.2e}] "
              f"({r['fails']} fails){extra}")
        print(f"  {'':9s} latency us (MEASURED): {lat}")
    if "cis_overlap" in res:
        verdict = "consistent (95% CIs overlap)" if res["cis_overlap"] else "DIFFERENT (CIs disjoint)"
        print(f"  bposd vs matching: {verdict}")
    print(f"  wrote {TRACES_DIR}/bposd_latency_{res['tag']}.npz/.json")


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--all", action="store_true", help="every workload in the grid, not just d5 p1e-3")
    ap.add_argument("--shots", type=int, default=None, help="use only the first N shots")
    ap.add_argument("--code", choices=("surface", "gross"))
    ap.add_argument("--max-distance", type=int, help="surface distance cap; retains gross")
    args = ap.parse_args(argv)

    grid = QUICK_GRID if args.grid == "quick" else FULL_GRID
    try:
        grid = select_workloads(grid, code=args.code, max_distance=args.max_distance, shots=args.shots)
    except ValueError as e:
        ap.error(str(e))
    if not args.all:
        grid = [w for w in grid if w.code == "surface" and w.distance == 5 and w.p == 1e-3]
    io.ensure_dirs()
    print(f"cpu: {measurement_info()['cpu']}  threads: 1  "
          f"(BP+OSD: max_iters={MAX_ITERS}, {BP_METHOD}, alpha={ALPHA}, {OSD_METHOD} order {OSD_ORDER})")
    results = []
    for w in grid:
        res = check(w, args.shots)
        _print(res)
        results.append(res)
    return results


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"p1_check_baselines: missing input ({e}); run `make data` first")
        sys.exit(1)
