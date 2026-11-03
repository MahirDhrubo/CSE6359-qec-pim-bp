# OWNER: P1
"""Print verified LER-vs-p numbers and iteration histograms as Markdown.

Run in the dataset working directory. Example: --grid full --shots 1000.
Missing/invalid traces fail unless --allow-partial is explicitly supplied.
This reports block LER for any logical Z failure (one surface logical, twelve
gross logicals), not per-round or per-logical-qubit error probability.
"""
import argparse

import numpy as np

from qdec import io
from qdec.contracts import FULL_GRID, FUNCTIONAL_COLS, QUICK_GRID, TRACES_DIR
from qdec.workload.selection import select_workloads
from qdec.workload.verification import verify_workload


def iteration_histogram(values):
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=51)
    if len(counts) != 51:
        raise ValueError("iteration count exceeds configured maximum of 50")
    return {"0": int(counts[0]), "1–5": int(counts[1:6].sum()),
            "6–10": int(counts[6:11].sum()), "11–25": int(counts[11:26].sum()),
            "26–49": int(counts[26:50].sum()), "50": int(counts[50])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("quick", "full"), default="full")
    ap.add_argument("--shots", type=int)
    ap.add_argument("--code", choices=("surface", "gross"))
    ap.add_argument("--max-distance", type=int)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args(argv)
    base = FULL_GRID if args.grid == "full" else QUICK_GRID
    grid = select_workloads(base, code=args.code, max_distance=args.max_distance, shots=args.shots)
    rows, missing = [], []
    for w in grid:
        try:
            result = verify_workload(w)
            df = io.read_table(TRACES_DIR / f"functional_{w.tag}.parquet", FUNCTIONAL_COLS)
            result["histogram"] = iteration_histogram(df.bp_iters)
            rows.append(result)
        except (OSError, ValueError, KeyError) as e:
            missing.append((w.tag, str(e)))
    if missing and not args.allow_partial:
        raise SystemExit("Incomplete or invalid sweep; use --allow-partial for a labeled progress report")
    print("# P1 Day 3 sweep report\n")
    print(f"{'REDUCED PROFILE' if grid != base else 'FULL REQUESTED GRID'}; "
          f"{'PARTIAL' if missing else 'complete for declared profile'}: "
          f"{len(rows)}/{len(grid)} workloads.\n")
    print("Block LER = any logical Z failure; gross has 12 logicals, surface has one. "
          "CIs are 95% Wilson; paired p tests quantized worse than float. "
          "A passing screen is not proof of equivalence. Small shot counts limit rare-event "
          "and p99.9 claims. Code distance labels are not measured circuit-level distances.\n")
    print("| Tag | N | Quant failures / LER [CI] | Float failures / LER [CI] | "
          "Paired p | LER screen | OSD rate | p99 iters |")
    print("|---|---:|---|---|---:|---|---:|---:|")
    for r in rows:
        def cell(prefix, r=r):
            lo, hi = r[f"{prefix}_ci"]
            return f"{r[f'{prefix}_fails']} / {r[f'{prefix}_ler']:.5g} [{lo:.5g}, {hi:.5g}]"
        print(f"| {r['tag']} | {r['n']} | {cell('quant')} | {cell('float')} | "
              f"{r['p_worse']:.5g} | {'PASS' if r['ler_screen_pass'] else 'FAIL'} | "
              f"{r['quant_osd_frac']:.2%} | {r['iters_p99']:g} |")
    print("\nIteration histograms (counts, including the zero-iteration path):\n")
    for r in rows:
        assert sum(r["histogram"].values()) == r["n"]
        print(f"- {r['tag']}: " + "; ".join(f"{k}: {v}" for k, v in r["histogram"].items()))
    if missing:
        print("\nMissing/invalid workloads:\n")
        for tag, error in missing:
            print(f"- {tag}: {error}")
    return rows


if __name__ == "__main__":
    main()
