# OWNER: P1
"""Proposed custom-OSD-only benchmark for P2 review; prints JSON, writes no files.

Usage: VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/p1_benchmark_osd.py --d 5 --p .001
Default: first 1,000 TEST shots, timings conditional on quantized BP fallback.
This is NOT ldpc timing, PIM latency, an LER experiment, or an adopted cost model.
No shared schema changes. P2 must agree the implementation and approximation first.
"""
import argparse
import json
import os
import time

import numpy as np

from qdec import io
from qdec.contracts import GRAPHS_DIR, QUICK_GRID, SHOTS_DIR
from qdec.workload.baselines import ALPHA, MAX_ITERS, OSD_METHOD, OSD_ORDER, measurement_info
from qdec.workload.bp_minsum import MinSumBP, channel_llrs
from qdec.workload.functional import LLR_RANGE_X, QUANT_BITS
from qdec.workload.osd import OSD


def time_fallbacks(bp, osd, dets, chunk=128):
    """Return (shot_id, elapsed_ns) only for fallback calls, after one untimed warm-up."""
    ids, timings = [], []
    warmed = False
    for start in range(0, len(dets), chunk):
        batch = dets[start:start + chunk]
        _, _, converged, llrs = bp.decode_batch(batch, return_llrs=True)
        for i in np.flatnonzero(~converged):
            if not warmed:
                osd.decode(batch[i], llrs[i])
                warmed = True
            t0 = time.perf_counter_ns()
            osd.decode(batch[i], llrs[i])
            timings.append(time.perf_counter_ns() - t0)
            ids.append(start + int(i))
    return np.asarray(ids, dtype=np.int64), np.asarray(timings, dtype=np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--p", type=float, default=.001)
    ap.add_argument("--shots", type=int, default=1000)
    args = ap.parse_args(argv)
    matches = [w for w in QUICK_GRID if w.distance == args.d and w.p == args.p]
    if len(matches) != 1 or not 0 < args.shots <= matches[0].shots:
        ap.error("choose a QUICK_GRID workload and 1..20,000 shots")
    w = matches[0]
    g = io.load_npz(GRAPHS_DIR / f"{w.code}_d{w.distance}_p{w.p:g}.npz")
    s = io.load_npz(SHOTS_DIR / f"{w.tag}.npz")
    if len(s["dets"]) < args.shots:
        raise ValueError("shot file is shorter than requested benchmark")
    bp = MinSumBP(g["H"], g["priors"], MAX_ITERS, ALPHA, QUANT_BITS,
                 llr_max=LLR_RANGE_X * channel_llrs(g["priors"]).max())
    decoder = OSD(g["H"], g["priors"], OSD_ORDER, OSD_METHOD)
    ids, ns = time_fallbacks(bp, decoder, s["dets"][:args.shots])
    info = measurement_info()
    info.update(label="MEASURED custom Python OSD only; proposal, not adopted by P2",
                timer="perf_counter_ns around OSD.decode only, one untimed warm-up",
                threads="single Python caller; BLAS controlled by listed environment",
                thread_environment={k: os.environ.get(k, "unset") for k in
                                    ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
                                     "OMP_NUM_THREADS")},
                tag=w.tag, shots=args.shots, fallback_calls=len(ids), quant_bits=QUANT_BITS,
                range_x=LLR_RANGE_X, alpha=ALPHA, max_iters=MAX_ITERS,
                osd_method=OSD_METHOD, osd_order=OSD_ORDER,
                latency_ns_percentiles=({str(q): float(np.percentile(ns, q))
                                         for q in (50, 99, 99.9)} if len(ns) else None))
    print(json.dumps(info, indent=2))
    return ids, ns


if __name__ == "__main__":
    main()
