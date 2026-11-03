# OWNER: P1
"""Print a reproducible Markdown TEST report without modifying functional artifacts.

Usage: .venv/bin/python scripts/p1_report_day2.py
Input hashes identify inspected artifacts, not their original creation provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path

from qdec import io
from qdec.contracts import FUNCTIONAL_COLS, QUICK_GRID, SHOTS_DIR, TRACES_DIR
from qdec.workload.functional import LLR_RANGE_X, QUANT_BITS
from qdec.workload.reporting import summarize_functional


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, default=TRACES_DIR)
    args = ap.parse_args(argv)
    summaries, hashes = [], []
    for w in QUICK_GRID:
        trace = args.directory / f"functional_{w.tag}.parquet"
        shots = SHOTS_DIR / f"{w.tag}.npz"
        sidecar = trace.with_suffix(".json")
        metadata = {}
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text())
            if (metadata["kind"] != "FUNCTIONAL"
                    or metadata["output_sha256"] != hashlib.sha256(trace.read_bytes()).hexdigest()
                    or metadata["input_sha256"][str(shots)]
                    != hashlib.sha256(shots.read_bytes()).hexdigest()
                    or metadata["decoder_settings"]["quant_bits"] != QUANT_BITS
                    or metadata["decoder_settings"]["range_x"] != LLR_RANGE_X):
                raise ValueError(f"{sidecar}: provenance does not match selected run")
        summaries.append(summarize_functional(
            w, io.read_table(trace, FUNCTIONAL_COLS), io.load_npz(shots), QUANT_BITS))
        summaries[-1]["float_osd_frac"] = metadata.get("float_diagnostics", {}).get("osd_frac")
        hashes.extend((str(p), hashlib.sha256(p.read_bytes()).hexdigest()) for p in (trace, shots))
    print("# P1 Day 2 TEST comparison\n")
    print(f"Selected configuration: {QUANT_BITS}-bit messages, range x{LLR_RANGE_X:g}. "
          "For legacy traces without metadata, range provenance is not certified.\n")
    print("Float = our min-sum BP + our OSD; CPU timing baseline = ldpc BP+OSD.\n")
    print("PASS means no significant degradation detected, not proven equivalence. "
          "All intervals are 95% Wilson; p-values are one-sided paired McNemar. "
          "TEST results were inspected during earlier iterations; this is not a blinded study.\n")
    print("| Workload | Bits | N | Quant failures / LER [CI] | Float failures / LER [CI] | "
          "Matching failures | Only quant / only float | p | LER screen |")
    print("|---|---:|---:|---|---|---:|---:|---:|---|")
    for r in summaries:
        def cell(prefix, r=r):
            lo, hi = r[f"{prefix}_ci"]
            return (f"{r[f'{prefix}_fails']} / {r[f'{prefix}_ler']:.6g} "
                    f"[{lo:.6g}, {hi:.6g}]")
        print(f"| {r['tag']} | {r['quant_bits']} | {r['n']} | {cell('quant')} | "
              f"{cell('float')} | {r['matching_fails']} | "
              f"{r['only_q_fails']} / {r['only_float_fails']} | {r['p_worse']:.6g} | "
              f"{'PASS' if r['ler_screen_pass'] else 'FAIL'} |")
    print("\n| Workload | Quantized OSD rate | Float OSD rate | OSD ratio | "
          "Quantized mean iterations | p99 | p99.9 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in summaries:
        f = r["float_osd_frac"]
        f_text = "unavailable" if f is None else f"{f:.3%}"
        ratio = f"{r['quant_osd_frac'] / f:.4f}" if f else "n/a"
        print(f"| {r['tag']} | {r['quant_osd_frac']:.3%} | {f_text} | {ratio} | "
              f"{r['iters_mean']:.4f} | "
              f"{r['iters_p99']:g} | {r['iters_p999']:g} |")
    print("\nFloat OSD rates come only from verified generation sidecars; "
          "legacy traces without those diagnostics report unavailable.")
    print("\nInspected artifact SHA-256 (does not certify original generation settings):\n")
    for path, digest in hashes:
        print(f"- `{path}`: `{digest}`")
    return summaries


if __name__ == "__main__":
    main()
