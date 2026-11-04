# OWNER: P2
"""Measure the CPU OSD fallback cost per workload from paired ldpc timings on the same shots.

For TEST shots where BP does not converge, time ldpc's BpOsdDecoder (BP + OSD) and BpDecoder
(BP only) with identical settings on the same syndrome, interleaved, best of --repeats, and take
OSD = t(BP+OSD) - t(BP) per shot. This pairs the two timings on one syndrome, so it is a
conditional per-call estimate; it is not the subtraction of medians from different shot
populations that P1's DAY2_HANDOFF rules out.
Settings come from P1's measured hand-off TRACES_DIR/bposd_latency_<tag>.json, so the OSD is the
one inside the bposd_cpu baseline. MEASURED on this CPU, one thread: run on an idle machine.
Writes COSTS_DIR/osd_<tag>.json (P2-internal; read by p2_build_costs.py for osd_ns).
Usage: python scripts/p2_measure_osd.py [--grid quick|full] [--max-shots N] [--repeats R]
"""
import argparse
import json
import platform
import time
from importlib.metadata import version

import numpy as np
import scipy.sparse as sp
from ldpc import BpDecoder, BpOsdDecoder

from qdec import io
from qdec.contracts import (
    COSTS_DIR,
    FULL_GRID,
    GRAPHS_DIR,
    QUICK_GRID,
    SEED_TEST,
    SHOTS_DIR,
    TRACES_DIR,
)

PCTS = (50, 90, 99, 99.9)


def _stats(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), **{f"p{q:g}": float(np.percentile(x, q)) for q in PCTS},
            "max": float(x.max())}


def measure(w, max_shots: int, repeats: int) -> dict:
    settings = json.loads((TRACES_DIR / f"bposd_latency_{w.tag}.json").read_text())
    g = io.load_npz(GRAPHS_DIR / f"{w.code}_d{w.distance}_p{w.p:g}.npz")
    dets = np.ascontiguousarray(io.load_npz(SHOTS_DIR / f"{w.tag}.npz")["dets"], dtype=np.uint8)
    H = sp.csr_matrix(g["H"], dtype=np.uint8)
    bp_kw = {"error_channel": list(np.asarray(g["priors"], dtype=np.float64)),
             "max_iter": settings["max_iters"], "bp_method": settings["bp_method"],
             "ms_scaling_factor": settings["alpha"], "omp_thread_count": 1}
    bp = BpDecoder(H, **bp_kw)
    bposd = BpOsdDecoder(H, osd_method=settings["osd_method"], osd_order=settings["osd_order"], **bp_kw)

    # untimed pass: which shots fall back to OSD (BP does not converge on a non-empty syndrome)
    osd_ids = []
    for i in np.flatnonzero(dets.any(axis=1)):
        bp.decode(dets[i])
        if not bp.converge:
            osd_ids.append(i)
    osd_ids = np.asarray(osd_ids, dtype=np.int64)
    rng = np.random.default_rng(SEED_TEST)
    timed = np.sort(rng.choice(osd_ids, min(max_shots, len(osd_ids)), replace=False))

    clock = time.perf_counter_ns
    bp.decode(dets[timed[0]])            # warm-up, not timed
    bposd.decode(dets[timed[0]])
    t_bp = np.full(len(timed), np.inf)
    t_all = np.full(len(timed), np.inf)
    mismatch = 0
    for r in range(repeats):
        for k, i in enumerate(timed):
            first, second = (bp, bposd) if (k + r) % 2 else (bposd, bp)   # alternate order
            for dec in (first, second):
                t0 = clock()
                dec.decode(dets[i])
                dt = clock() - t0
                if dec is bp:
                    t_bp[k] = min(t_bp[k], dt)
                else:
                    t_all[k] = min(t_all[k], dt)
                    mismatch += r == 0 and bool(dec.converge)
    osd = t_all - t_bp
    return {
        "label": "MEASURED-derived (paired per shot: ldpc BP+OSD minus ldpc BP on the same syndrome)",
        "tag": w.tag, "osd_ns": _stats(osd), "bp_only_ns": _stats(t_bp), "bp_osd_ns": _stats(t_all),
        "osd_frac_negative": float((osd < 0).mean()),
        "n_test_shots": len(dets), "n_osd_shots": len(osd_ids), "n_timed": len(timed),
        "repeats": repeats, "converge_mismatch": int(mismatch),
        "settings": {k: settings[k] for k in ("max_iters", "osd_order", "bp_method", "osd_method", "alpha")},
        "cpu": platform.processor() or platform.machine(), "threads": 1,
        "platform": platform.platform(), "ldpc": version("ldpc"),
        "timer": "time.perf_counter_ns around decoder.decode, best of repeats, after one warm-up",
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--max-shots", type=int, default=2000, help="OSD shots timed per workload")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args(argv)

    io.ensure_dirs()
    for w in QUICK_GRID if args.grid == "quick" else FULL_GRID:
        try:
            res = measure(w, args.max_shots, args.repeats)
        except FileNotFoundError as e:
            print(f"{w.tag}: SKIPPED, missing {e.filename} (run make data and p1_check_baselines --all)")
            continue
        path = COSTS_DIR / f"osd_{w.tag}.json"
        path.write_text(json.dumps(res, indent=2))
        o = res["osd_ns"]
        print(f"{w.tag}: OSD on {res['n_osd_shots']}/{res['n_test_shots']} shots, timed "
              f"{res['n_timed']}: median {o['p50'] / 1e3:,.1f} us  p99 {o['p99'] / 1e3:,.1f} us  "
              f"(BP-only median {res['bp_only_ns']['p50'] / 1e3:,.1f} us, "
              f"converge mismatch {res['converge_mismatch']})  -> {path}", flush=True)


if __name__ == "__main__":
    main()
