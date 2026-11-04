# OWNER: P2
"""Measure the CPU OSD fallback cost per workload from paired ldpc timings on the same shots.

See qdec.costmodel.osd_timing for the method (OSD = t(BP+OSD) - t(BP) on one non-converged
TEST syndrome). Decoder settings come from P1's TRACES_DIR/bposd_latency_<tag>.json when present
(so the OSD is the one inside bposd_cpu), else from osd_timing.REFERENCE_SETTINGS (P1's published
hand-off values). MEASURED on this CPU, one thread: run on an otherwise idle machine.
Writes COSTS_DIR/osd_<tag>.json (P2-internal; read by p2_build_costs.py for osd_ns).
p2_build_costs.py also calls write_osd() itself for any tag without a measurement.
Usage: python scripts/p2_measure_osd.py [--grid quick|full] [--max-shots N] [--repeats R]
                                        [--budget-s S]
"""
import argparse

from qdec import io
from qdec.contracts import FULL_GRID, QUICK_GRID
from qdec.costmodel.osd_timing import summary, write_osd


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--max-shots", type=int, default=2000, help="OSD shots timed per workload")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--budget-s", type=float, default=None, help="seconds per workload (default: none)")
    args = ap.parse_args(argv)

    io.ensure_dirs()
    for w in QUICK_GRID if args.grid == "quick" else FULL_GRID:
        try:
            path, res = write_osd(w, args.max_shots, args.repeats, args.budget_s)
        except FileNotFoundError as e:
            print(f"{w.tag}: SKIPPED, missing {e.filename} (run make data)")
            continue
        print(f"{summary(res)}  -> {path}", flush=True)


if __name__ == "__main__":
    main()
