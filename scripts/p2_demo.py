# OWNER: P2
"""P2 live demo: syndrome -> clusters -> cache hit / BP fallback -> service time.

Self-contained, so it runs from a fresh clone in well under 2 minutes with no data/ files: it
samples its own SEED_TRAIN / SEED_TEST shots for one surface-code workload with Stim, builds the
DecodeCache on TRAIN only, decodes every TEST shot with ldpc BP+OSD (timed live, = bposd_cpu),
times the OSD fallback live (paired, as in osd_timing), and prices each shot on all 7 systems
with the same cost model as the result tables (costmodel.service.build_service_table).
Then it walks through example shots (a cache hit, a BP miss, an OSD miss):
  1. the syndrome on the lattice, one panel per round; each defect shows its cluster's letter,
     UPPERCASE if that cluster can be served from the cache, lowercase if not,
  2. each cluster's size and why it hits or misses,
  3. the path taken and whether the correction is logically right,
  4. that shot's service time on every system (MODELED / PUBLISHED / MEASURED labels).
Live BP iteration counts come from ldpc's float min-sum with P1's settings; the result tables
use P1's 6-bit quantized traces instead.
Also writes the demo's service table (SERVICE_COLS) to data/p2/demo/, for P3's queue demo.
Usage: python scripts/p2_demo.py [--d 5] [--p 0.001] [--train-shots N] [--test-shots N]
"""
import argparse
import string
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

T0 = time.perf_counter()

from qdec import io
from qdec.cache.cam import DEFAULT_CAM_ENTRIES, ClusterCache
from qdec.cache.devshots import sample_dev_shots
from qdec.contracts import DATA, GRAPH_STATS_KEYS, SEED_TEST, SEED_TRAIN, SERVICE_COLS, System
from qdec.costmodel import pim
from qdec.costmodel.osd_timing import REFERENCE_SETTINGS, decoders, measure_osd
from qdec.costmodel.service import build_service_table, load_costs

QUANT_BITS = 6   # P1's frozen message width (workload/functional.py); sets the modeled width only
ORDER = [System.BPOSD_CPU, System.BP_CPU_CONV, System.NEARMEM_DPU, System.RELAY_FPGA,
         System.PIM_NOCACHE, System.PIM_CACHE, System.FULL]
LABEL = {System.BPOSD_CPU: "MEASURED live", System.NEARMEM_DPU: "PUBLISHED",
         System.RELAY_FPGA: "PUBLISHED"}
NAME = {System.FULL: "full (DecodeCache + PIM-BP)", System.PIM_CACHE: "pim_cache (no controller)"}
OUT = DATA / "p2" / "demo"


def stats_of(H) -> dict:
    """GRAPH_STATS_KEYS for the demo's own graph (same definitions as P1's graph_stats)."""
    H = sp.csr_matrix(H)
    s = {"n_vars": H.shape[1], "n_checks": H.shape[0], "n_edges": H.nnz,
         "max_var_deg": int(H.getnnz(axis=0).max()), "max_check_deg": int(H.getnnz(axis=1).max())}
    return {k: int(s[k]) for k in GRAPH_STATS_KEYS}


def us(ns: float) -> str:
    return "n/a" if np.isnan(ns) else f"{ns / 1e3:,.2f} us" if ns < 1e6 else f"{ns / 1e6:,.2f} ms"


def lattice(cache: ClusterCache, split, status: list[str], max_panels: int = 6) -> list[str]:
    """One text panel per round that has defects; cell = '.' detector, letter = defect."""
    g = cache._grid
    letter = {}
    for n, (c, st) in enumerate(zip(split.clusters, status)):
        ch = string.ascii_uppercase[n % 26]
        for d in c.det_ids:
            letter[d] = ch if st == "hit" else ch.lower()
    rounds = sorted({int(g[d, 2]) for d in letter})[:max_panels]
    X, Y = int(g[:, 0].max()), int(g[:, 1].max())
    where = {(int(x), int(y), int(t)): i for i, (x, y, t) in enumerate(g)}
    panels = []
    for t in rounds:
        rows = [f"t={t}".ljust(2 * (X + 1))]
        for y in range(Y + 1):
            rows.append("".join(
                (letter.get(where[(x, y, t)], ".") if (x, y, t) in where else " ") + " "
                for x in range(X + 1)))
        panels.append(rows)
    return ["   ".join(r[i] for r in panels) for i in range(Y + 2)] if panels else ["(no defects)"]


def show_shot(title, i, cache, split, status, row_f, row_c, svc, fail_bposd, per_pim):
    print(f"\n--- {title}: TEST shot {i}, {len(split.clusters)} cluster(s), "
          f"{row_f.n_defects} defect(s) ---")
    for line in lattice(cache, split, status):
        print("   " + line)
    print("   cluster  size  served?")
    for n, (c, st) in enumerate(zip(split.clusters, status)):
        why = "yes (isolated, key in CAM)" if st == "hit" else f"no ({st})"
        print(f"   {string.ascii_uppercase[n % 26]:7s}  {len(c.det_ids):4d}  {why}")
    if row_c.cache_hit:
        ok = "correct" if not row_c.fail_cache else "LOGICAL ERROR"
        print(f"   => CACHE HIT: {row_c.n_lookups} CAM search(es), stored corrections replayed; "
              f"logical result: {ok}")
        if row_f.bp_iters:
            print(f"      (without the cache this shot needs {row_f.bp_iters} BP iterations"
                  f"{' + OSD' if row_f.osd_ran else ''})")
    else:
        bp = f"BP converged in {row_f.bp_iters} iterations" if not row_f.osd_ran else \
             f"BP did not converge in {row_f.bp_iters} iterations -> OSD on the CPU"
        ok = "correct" if not fail_bposd[i] else "LOGICAL ERROR"
        print(f"   => CACHE MISS -> {bp}; logical result: {ok}")
    print("   service time for this shot:")
    rows = svc[svc["shot_id"] == i].set_index("system")
    for s in ORDER:
        if s.value in rows.index:
            r = rows.loc[s.value]
            extra = f"  ({row_f.bp_iters} x {per_pim / 1e3:.2f} us per iteration)" \
                if s == System.PIM_NOCACHE and row_f.bp_iters else ""
            print(f"     {NAME.get(s, s.value):28s} {r['path']:7s} {us(r['latency_ns']):>12s}  "
                  f"{LABEL.get(s, 'MODELED')}{extra}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--p", type=float, default=1e-3)
    ap.add_argument("--train-shots", type=int, default=20_000)
    ap.add_argument("--test-shots", type=int, default=2_000)
    ap.add_argument("--osd-budget-s", type=float, default=8.0, help="live OSD timing budget")
    args = ap.parse_args(argv)
    d, p = args.d, args.p
    train_tag, test_tag = f"surface_d{d}_p{p:g}_s{SEED_TRAIN}", f"surface_d{d}_p{p:g}_s{SEED_TEST}"

    print(f"== DecodeCache + PIM-BP demo: rotated surface code d={d}, p={p:g}, circuit-level noise ==")
    train = sample_dev_shots(d, p, args.train_shots, SEED_TRAIN)
    test = sample_dev_shots(d, p, args.test_shots, SEED_TEST)
    cache = ClusterCache(train, train["det_coords"], max_entries=DEFAULT_CAM_ENTRIES)
    cache.build(train, train_tag)
    print(f"cache built on {args.train_shots:,} TRAIN shots (seed {SEED_TRAIN}): {cache.size():,} CAM "
          f"entries x {cache.key_bits()} bits  [{time.perf_counter() - T0:.0f} s]")

    splits = cache.split(test)
    cdf = cache.evaluate(test, test_tag, splits)

    # ldpc BP+OSD on every TEST shot: live bposd_cpu latency, iteration counts, logical failures
    _bp, bposd = decoders(test["H"], test["priors"], REFERENCE_SETTINGS)
    dets = np.ascontiguousarray(test["dets"], dtype=np.uint8)
    n, max_iters = len(dets), REFERENCE_SETTINGS["max_iters"]
    iters, osd = np.zeros(n, np.int32), np.zeros(n, bool)
    lat, corr = np.zeros(n), np.zeros((n, test["H"].shape[1]), np.uint8)
    bposd.decode(dets[0])                                   # warm-up
    for i in range(n):
        t0 = time.perf_counter_ns()
        corr[i] = bposd.decode(dets[i])
        lat[i] = time.perf_counter_ns() - t0
        if dets[i].any():
            osd[i] = not bposd.converge
            iters[i] = max_iters if osd[i] else bposd.iter
    pred = (sp.csr_matrix(test["L"], dtype=np.uint8) @ corr.T % 2).T.astype(bool)
    fail_bposd = (pred != test["obs"]).any(axis=1)
    osd_res = measure_osd(test["H"], test["priors"], test["dets"], REFERENCE_SETTINGS,
                          max_shots=200, repeats=1, budget_s=args.osd_budget_s)
    print(f"decoded {n:,} TEST shots (seed {SEED_TEST}) with ldpc BP+OSD; OSD fallback measured "
          f"live on {osd_res['n_timed']} shots: median {us(osd_res['osd_ns']['p50'])}  "
          f"[{time.perf_counter() - T0:.0f} s]")

    f = pd.DataFrame({"shot_id": np.arange(n), "tag": test_tag, "quant_bits": QUANT_BITS,
                      "n_defects": dets.sum(axis=1), "bp_iters": iters, "osd_ran": osd})
    stats, costs = stats_of(test["H"]), load_costs()
    svc = build_service_table(f, cdf, stats, costs, cam_entries=cache.size(),
                              key_bits=cache.key_bits(), bposd_latency_ns=lat,
                              osd_ns=osd_res["osd_ns"]["p50"])
    per_pim = pim.per_iteration_cost(stats, QUANT_BITS, costs)["ns"]

    hit = cdf["cache_hit"].to_numpy()
    nclu = cdf["n_clusters"].to_numpy()
    picks = [("CACHE HIT", np.flatnonzero(hit & (nclu >= 2)), np.flatnonzero(hit & (nclu >= 1))),
             ("MISS -> BP", np.flatnonzero(~hit & ~osd & (nclu >= 2)), np.flatnonzero(~hit & ~osd)),
             ("MISS -> BP + OSD", np.flatnonzero(~hit & osd), np.array([], int))]
    for title, best, fallback in picks:
        ids = best if len(best) else fallback
        if len(ids):
            i = int(ids[0])
            show_shot(title, i, cache, splits[i], cache.cluster_status(splits[i]), f.iloc[i],
                      cdf.iloc[i], svc, fail_bposd, per_pim)

    nonempty = nclu > 0
    print(f"\n=== all {n:,} TEST shots ===")
    print(f"cache hit rate: {hit.mean():.1%} of shots, {hit[nonempty].mean():.1%} of shots with "
          f"defects; logical errors on hit shots: cache {int(cdf['fail_cache'].sum())}, "
          f"BP+OSD {int(fail_bposd[hit].sum())} (of {int(hit.sum()):,})")
    print(f"OSD fallback on {osd.mean():.1%} of shots at {us(osd_res['osd_ns']['p50'])} each "
          f"(MEASURED live, paired ldpc timing)")
    print(f"{'system':28s} {'label':14s} {'p50':>11s} {'p99':>11s} {'mean':>11s} {'mean energy':>12s}")
    for s in ORDER:
        g = svc[svc["system"] == s.value]
        e = g["energy_nj"].mean()
        print(f"{NAME.get(s, s.value):28s} {LABEL.get(s, 'MODELED'):14s} "
              f"{us(g['latency_ns'].median()):>11s} {us(g['latency_ns'].quantile(0.99)):>11s} "
              f"{us(g['latency_ns'].mean()):>11s} "
              f"{'n/a' if np.isnan(e) else f'{e / 1e3:,.1f} uJ':>12s}")
    print("full = pim_cache here: the backlog controller is applied in P3's queue demo.")
    path = io.write_table(svc, SERVICE_COLS, OUT / f"service_{test_tag}.parquet")
    print(f"service table for P3's queue demo: {path}")
    print(f"demo finished in {time.perf_counter() - T0:.0f} s")


if __name__ == "__main__":
    main()
