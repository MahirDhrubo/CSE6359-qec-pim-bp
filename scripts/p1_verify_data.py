# OWNER: P1
"""Verify P1 shots, graphs and real functional traces for a declared profile.

Run from the dataset working directory (which contains data/).
Exit nonzero on missing, stale, mock or structurally invalid artifacts. Reduced
profiles are explicitly labeled; --data-only never certifies functional output.
No imports from P2/P3 and no changes to shared contracts.
"""
import argparse
import os
from pathlib import Path

from qdec.contracts import FULL_GRID, QUICK_GRID
from qdec.workload.selection import select_workloads
from qdec.workload.verification import verify_workload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--code", choices=("surface", "gross"))
    ap.add_argument("--max-distance", type=int)
    ap.add_argument("--shots", type=int)
    ap.add_argument("--data-only", action="store_true")
    ap.add_argument("--require-latency", action="store_true")
    ap.add_argument("--work-dir", type=Path, help="dataset directory containing data/")
    args = ap.parse_args(argv)
    if args.work_dir is not None:
        os.chdir(args.work_dir)
    base = QUICK_GRID if args.grid == "quick" else FULL_GRID
    selected = select_workloads(base, code=args.code, max_distance=args.max_distance, shots=args.shots)
    label = "COMPLETE GRID" if selected == base else "REDUCED PROFILE"
    print(f"{label}: {len(selected)}/{len(base)} workloads, "
          f"{sum(w.shots for w in selected)} TEST shots; "
          f"{'DATA ONLY' if args.data_only else 'P1 functional artifacts'}")
    errors, results = [], []
    for w in selected:
        try:
            r = verify_workload(w, data_only=args.data_only, require_latency=args.require_latency)
            results.append(r)
            print(f"OK {w.tag} ({w.shots} shots)")
        except (OSError, ValueError, KeyError) as e:
            errors.append((w.tag, str(e)))
            print(f"FAIL {w.tag}: {e}")
    if errors:
        raise SystemExit(1)
    print("VERIFIED declared profile. This does not certify P2/P3 outputs or LER equivalence.")
    return results


if __name__ == "__main__":
    main()
