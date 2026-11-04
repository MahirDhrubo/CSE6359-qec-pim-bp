# OWNER: P2
"""CPU OSD fallback cost from paired ldpc timings on the same shots. MEASURED.

For a syndrome where BP does not converge, time ldpc's BpOsdDecoder (BP + OSD) and BpDecoder
(BP only) with identical settings on that same syndrome, alternating order, best of `repeats`,
and take OSD = t(BP+OSD) - t(BP). Pairing on one syndrome gives a conditional per-call estimate;
it is not the subtraction of medians from different shot populations that P1's DAY2_HANDOFF
rules out. Single thread; the numbers describe this CPU and ldpc's OSD-CS, not hardware PIM.
"""
import json
import platform
import time
from importlib.metadata import version

import numpy as np
import scipy.sparse as sp
from ldpc import BpDecoder, BpOsdDecoder

from qdec import io
from qdec.contracts import COSTS_DIR, GRAPHS_DIR, SEED_TEST, SHOTS_DIR, TRACES_DIR

# P1's bposd_cpu decoder settings (src/qdec/workload/DAY2_HANDOFF.md). P1's measured hand-off
# TRACES_DIR/bposd_latency_<tag>.json overrides them when present, so the OSD timed here is the
# one inside the bposd_cpu baseline.
REFERENCE_SETTINGS = {"max_iters": 50, "osd_order": 7, "bp_method": "minimum_sum",
                      "osd_method": "osd_cs", "alpha": 0.625}
PCTS = (50, 90, 99, 99.9)


def decoders(H, priors, settings: dict) -> tuple[BpDecoder, BpOsdDecoder]:
    """BP-only and BP+OSD ldpc decoders with identical BP settings, one thread each."""
    H = sp.csr_matrix(H, dtype=np.uint8)
    kw = {"error_channel": list(np.asarray(priors, dtype=np.float64)),
          "max_iter": settings["max_iters"], "bp_method": settings["bp_method"],
          "ms_scaling_factor": settings["alpha"], "omp_thread_count": 1}
    return (BpDecoder(H, **kw),
            BpOsdDecoder(H, osd_method=settings["osd_method"], osd_order=settings["osd_order"], **kw))


def paired_time(bp: BpDecoder, bposd: BpOsdDecoder, syndrome: np.ndarray,
                repeats: int = 3, first: int = 0) -> tuple[float, float]:
    """(t_bp, t_bp_osd) in ns for one syndrome: best of `repeats`, decoder order alternating."""
    clock = time.perf_counter_ns
    best = {id(bp): np.inf, id(bposd): np.inf}
    for r in range(repeats):
        for dec in ((bp, bposd) if (first + r) % 2 else (bposd, bp)):
            t0 = clock()
            dec.decode(syndrome)
            best[id(dec)] = min(best[id(dec)], clock() - t0)
    return float(best[id(bp)]), float(best[id(bposd)])


def _stats(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), **{f"p{q:g}": float(np.percentile(x, q)) for q in PCTS},
            "max": float(x.max())}


def measure_osd(H, priors, dets: np.ndarray, settings: dict | None = None, *,
                max_shots: int = 2000, repeats: int = 3, budget_s: float | None = None,
                min_shots: int = 20) -> dict:
    """Paired OSD timing on up to max_shots non-converged shots of `dets` (TEST shots).

    Shots are visited in a SEED_TEST-seeded random order. With budget_s, scanning for OSD shots
    and timing them stop once the budget is spent (after at least min_shots are timed), so large
    graphs, where one OSD call takes seconds, still finish. Returns the osd_<tag>.json payload
    (osd_ns percentiles, BP-only and BP+OSD percentiles, counts, settings, CPU provenance).
    """
    settings = settings or REFERENCE_SETTINGS
    bp, bposd = decoders(H, priors, settings)
    dets = np.ascontiguousarray(dets, dtype=np.uint8)
    start = time.perf_counter()

    def out_of_time(frac: float) -> bool:
        return budget_s is not None and time.perf_counter() - start > frac * budget_s

    order = np.random.default_rng(SEED_TEST).permutation(np.flatnonzero(dets.any(axis=1)))
    found, scanned = [], 0
    for i in order:                      # untimed scan: which shots fall back to OSD
        scanned += 1
        bp.decode(dets[i])
        if not bp.converge:
            found.append(int(i))
        if len(found) >= max_shots or (len(found) >= min_shots and out_of_time(1 / 3)):
            break
    if not found:
        raise ValueError("no shot fell back to OSD; nothing to time")

    paired_time(bp, bposd, dets[found[0]], repeats=1)   # warm-up, not timed
    t_bp, t_all, mismatch = [], [], 0
    for k, i in enumerate(found):
        a, b = paired_time(bp, bposd, dets[i], repeats, first=k)
        t_bp.append(a)
        t_all.append(b)
        mismatch += bool(bposd.converge)
        if len(t_bp) >= min_shots and out_of_time(1.0):
            break
    t_bp, t_all = np.asarray(t_bp), np.asarray(t_all)
    osd = t_all - t_bp
    est_osd_total = len(found) / max(scanned, 1) * len(order)
    return {
        "label": "MEASURED-derived (paired per shot: ldpc BP+OSD minus ldpc BP on the same syndrome)",
        "osd_ns": _stats(osd), "bp_only_ns": _stats(t_bp), "bp_osd_ns": _stats(t_all),
        "osd_frac_negative": float((osd < 0).mean()),
        "n_test_shots": len(dets), "n_scanned": int(scanned),
        "n_osd_shots_est": round(est_osd_total), "n_timed": len(osd),
        "repeats": repeats, "budget_s": budget_s, "converge_mismatch": int(mismatch),
        "settings": {k: settings[k] for k in REFERENCE_SETTINGS},
        "cpu": platform.processor() or platform.machine(), "threads": 1,
        "platform": platform.platform(), "ldpc": version("ldpc"),
        "timer": "time.perf_counter_ns around decoder.decode, best of repeats, after one warm-up",
    }


def write_osd(w, max_shots: int = 2000, repeats: int = 3, budget_s: float | None = None):
    """Measure one TEST workload and write COSTS_DIR/osd_<tag>.json; returns (path, payload)."""
    g = io.load_npz(GRAPHS_DIR / f"{w.code}_d{w.distance}_p{w.p:g}.npz")
    dets = io.load_npz(SHOTS_DIR / f"{w.tag}.npz")["dets"]
    p1 = TRACES_DIR / f"bposd_latency_{w.tag}.json"
    if p1.exists():
        settings, src = json.loads(p1.read_text()), p1.name
    else:
        settings, src = REFERENCE_SETTINGS, "osd_timing.REFERENCE_SETTINGS (P1 DAY2_HANDOFF)"
    res = {"tag": w.tag, "settings_source": src,
           **measure_osd(g["H"], g["priors"], dets, settings, max_shots=max_shots,
                         repeats=repeats, budget_s=budget_s)}
    path = COSTS_DIR / f"osd_{w.tag}.json"
    path.write_text(json.dumps(res, indent=2))
    return path, res


def summary(res: dict) -> str:
    o, b = res["osd_ns"], res["bp_only_ns"]
    return (f"{res['tag']}: timed {res['n_timed']} OSD shots (~{res['n_osd_shots_est']} of "
            f"{res['n_test_shots']} need OSD): median {o['p50'] / 1e3:,.1f} us  p99 "
            f"{o['p99'] / 1e3:,.1f} us  (BP-only median {b['p50'] / 1e3:,.1f} us, "
            f"converge mismatch {res['converge_mismatch']})")
