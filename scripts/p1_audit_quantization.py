# OWNER: P1
"""Print a TRAIN-only QUICK_GRID coverage audit; never writes or reruns a sweep.

Usage: .venv/bin/python scripts/p1_audit_quantization.py
PASS means overlapping Wilson intervals, paired p >= .05, and OSD <= tolerance
times float. It is not a proof of statistical equivalence. Missing measurements
are UNKNOWN, never PASS. Legacy CSVs lack settings/input hashes, so this audit
checks coverage but cannot certify their provenance.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from qdec.contracts import QUICK_GRID, SEED_TRAIN, TRACES_DIR
from qdec.workload.stats import cis_overlap, wilson_ci


def audit(directory: Path, bits=(3, 4, 5, 6, 7, 8), ranges=(1.0, 1.5, 2.0, 3.0),
          osd_tol=1.15) -> pd.DataFrame:
    """Require full TRAIN counts and one reference per file before comparing rows.

    Returns an in-memory diagnostic table, not a new cross-owner file schema.
    A config is globally rejected if any workload rejects it, even if others
    are missing; it is globally accepted only if all four workloads pass.
    """
    if not np.isfinite(osd_tol) or osd_tol < 1:
        raise ValueError("OSD tolerance must be finite and >= 1")
    rows = []
    for workload in QUICK_GRID:
        w = replace(workload, seed=SEED_TRAIN)
        path = directory / f"quant_sweep_{w.tag}.csv"
        df = pd.read_csv(path) if path.exists() else None
        ref = None
        if df is not None:
            if not df["tag"].eq(w.tag).all() or not df["n"].eq(w.shots).all():
                raise ValueError(f"{path}: expected {w.shots} TRAIN shots for {w.tag}")
            refs = df[df.quant_bits.isna()]
            if len(refs) != 1 or df.duplicated(["quant_bits", "range_x"]).any():
                raise ValueError(f"{path}: require one float reference and unique configs")
            for col, upper in (("fails", w.shots), ("osd_frac", 1), ("p_worse", 1)):
                if not (np.isfinite(df[col]) & df[col].between(0, upper)).all():
                    raise ValueError(f"{path}: invalid {col}")
            if not (df.fails == np.floor(df.fails)).all():
                raise ValueError(f"{path}: failures must be integer counts")
            ref = refs.iloc[0]
        for b in bits:
            for r in ranges:
                status, ratio = "MISSING", np.nan
                if df is not None:
                    match = df[(df.quant_bits == b) & (df.range_x == r)]
                    if len(match):
                        q = match.iloc[0]
                        ler_ok = (cis_overlap(wilson_ci(q.fails, w.shots),
                                              wilson_ci(ref.fails, w.shots))
                                  and q.p_worse >= .05)
                        osd_ok = q.osd_frac <= osd_tol * ref.osd_frac
                        ratio = (q.osd_frac / ref.osd_frac if ref.osd_frac else
                                 (1.0 if q.osd_frac == 0 else np.inf))
                        status = "PASS" if ler_ok and osd_ok else "FAIL"
                rows.append({"tag": w.tag, "bits": b, "range_x": r, "status": status,
                                 "osd_ratio": ratio})
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (bits, r), group in table.groupby(["bits", "range_x"], sort=True):
        statuses = group.status
        verdict = ("REJECTED" if statuses.eq("FAIL").any() else
                   "PASS ALL" if statuses.eq("PASS").all() else "INCOMPLETE")
        rows.append({"bits": bits, "range_x": r, "passed": int(statuses.eq("PASS").sum()),
                         "missing": int(statuses.eq("MISSING").sum()), "verdict": verdict})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, default=TRACES_DIR)
    ap.add_argument("--osd-tol", type=float, default=1.15)
    args = ap.parse_args(argv)
    table = audit(args.directory, osd_tol=args.osd_tol)
    print("TRAIN coverage (20,000 shots/workload; legacy provenance not certified)")
    print(table.pivot(index=["bits", "range_x"], columns="tag", values="status").to_string())
    summary = summarize(table)
    print("\nAcross QUICK_GRID:")
    print(summary.to_string(index=False))
    passing = summary[summary.verdict == "PASS ALL"]
    if passing.empty:
        print("\nNo configuration is supported at every workload yet.")
    else:
        smallest = int(passing.bits.min())
        unresolved = summary[(summary.bits < smallest) & (summary.verdict == "INCOMPLETE")]
        print(f"\nSmallest fully covered passing width in saved evidence: {smallest} bits.")
        print(f"Unresolved smaller-width configurations: {len(unresolved)}.")
    print("PASS means no significant degradation detected, not proven equivalence.")
    print("Candidate domain: 3–8 bits, ranges x1/x1.5/x2/x3; no claim outside this domain.")
    return table


if __name__ == "__main__":
    main()
