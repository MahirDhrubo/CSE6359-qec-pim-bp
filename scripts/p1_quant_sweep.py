# OWNER: P1
"""T11: quantized vs float min-sum BP+OSD LER sweep over quant_bits x message range.

The message range is llr_max = R x (largest channel LLR) for each R in --ranges. For every
(bits, R) we decode the same shots as the float reference and report LER with 95% Wilson CIs
plus an exact paired McNemar test (quantized fails more often than float?). A config passes
when the CIs overlap and the paired test is not significant (p >= 0.05). The pick is the
smallest passing quant_bits whose OSD rate stays within --osd-tol x float's: LER alone would
accept configs that send far more shots to the slow OSD path (4 bits: 68% vs 38% at d5).

Tune on TRAIN shots (default), then confirm the pick on TEST with --seed test --bits B
--ranges R, so the reported LER is not tuned on the shots it is measured on.
Writes TRACES_DIR/quant_sweep_<tag>.csv.
Usage: python scripts/p1_quant_sweep.py [--d 5] [--p 3e-3] [--seed train|test]
           [--bits 3,4,5,6,7,8] [--ranges 1,1.5,2,3] [--shots N] [--output-dir DIR]

Existing outputs are never overwritten. Use a new output directory for another run.
"""
import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qdec import io
from qdec.contracts import GRAPHS_DIR, QUICK_GRID, SEED_TEST, SEED_TRAIN, SHOTS_DIR, TRACES_DIR
from qdec.workload.baselines import ALPHA, MAX_ITERS, OSD_METHOD, OSD_ORDER, measurement_info
from qdec.workload.bp_minsum import channel_llrs
from qdec.workload.functional import minsum_bposd_batch
from qdec.workload.stats import cis_overlap, paired_worse_pvalue, wilson_ci


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def sweep(w, bits_list, ranges, shots=None, alpha=ALPHA) -> pd.DataFrame:
    g = io.load_npz(GRAPHS_DIR / f"{w.code}_d{w.distance}_p{w.p:g}.npz")
    s = io.load_npz(SHOTS_DIR / f"{w.tag}.npz")
    dets, obs = s["dets"][:shots], s["obs"][:shots]
    n = dets.shape[0]
    llr_top = float(channel_llrs(g["priors"]).max())
    common = {"alpha": alpha, "max_iters": MAX_ITERS, "osd_order": OSD_ORDER,
              "osd_method": OSD_METHOD}

    def run(bits, R):
        t0 = time.perf_counter()
        r = minsum_bposd_batch(g["H"], g["L"], g["priors"], dets, obs, quant_bits=bits,
                               llr_max=None if bits is None else R * llr_top, **common)
        return r, time.perf_counter() - t0

    ref, dt = run(None, None)
    rows = [_row(w, n, None, None, ref, ref, dt)]
    _show(rows[-1])
    for bits in bits_list:
        for R in ranges:
            r, dt = run(bits, R)
            rows.append(_row(w, n, bits, R, r, ref, dt))
            _show(rows[-1])
    return pd.DataFrame(rows)


def _row(w, n, bits, R, r, ref, dt) -> dict:
    k, kf = int(r["fail"].sum()), int(ref["fail"].sum())
    ci, ci_f = wilson_ci(k, n), wilson_ci(kf, n)
    n_qf, n_fq, p = paired_worse_pvalue(r["fail"], ref["fail"])
    it = r["bp_iters"]
    return {
        "tag": w.tag, "n": n, "quant_bits": bits, "range_x": R, "fails": k, "ler": k / n,
        "ci_lo": ci[0], "ci_hi": ci[1], "only_q_fails": n_qf, "only_float_fails": n_fq,
        "p_worse": p, "passes": bits is None or (cis_overlap(ci, ci_f) and p >= 0.05),
        "osd_frac": float(r["osd_ran"].mean()), "iters_mean": float(it.mean()),
        "iters_p99": float(np.percentile(it, 99)), "seconds": dt,
    }


def _show(r: dict) -> None:
    name = "float" if r["quant_bits"] is None else f"{r['quant_bits']}b x{r['range_x']:g}"
    print(f"  {name:10s} LER={r['ler']:.2e} [{r['ci_lo']:.2e},{r['ci_hi']:.2e}] "
          f"fails={r['fails']:4d} only_q/only_f={r['only_q_fails']}/{r['only_float_fails']} "
          f"p_worse={r['p_worse']:.3f} osd={r['osd_frac']:.1%} iters mean/p99="
          f"{r['iters_mean']:.1f}/{r['iters_p99']:.0f} {'PASS' if r['passes'] else 'fail'} "
          f"({r['seconds']:.0f}s)", flush=True)


# Team decision 2026-10-31 (T11): quantized OSD rate may be at most 1.15x the float one.
OSD_TOL = 1.15


def pick(df: pd.DataFrame, osd_tol: float = OSD_TOL) -> pd.Series | None:
    """Smallest quant_bits that passes on LER with OSD rate <= osd_tol x float's; among its
    ranges the lowest LER, then lowest OSD rate."""
    float_osd = df.loc[df["quant_bits"].isna(), "osd_frac"].iloc[0]
    q = df[df["quant_bits"].notna() & df["passes"] & (df["osd_frac"] <= osd_tol * float_osd)]
    if q.empty:
        return None
    q = q[q["quant_bits"] == q["quant_bits"].min()]
    return q.sort_values(["ler", "osd_frac"]).iloc[0]


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--p", type=float, default=3e-3)
    ap.add_argument("--seed", choices=("train", "test"), default="train")
    ap.add_argument("--bits", default="3,4,5,6,7,8")
    ap.add_argument("--ranges", default="1,1.5,2,3")
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--output-dir", type=Path, default=TRACES_DIR,
                    help="use a fresh directory to preserve earlier experiments")
    ap.add_argument("--osd-tol", type=float, default=OSD_TOL,
                    help="max quantized/float OSD-rate ratio for the pick")
    args = ap.parse_args(argv)

    matches = [w for w in QUICK_GRID if w.distance == args.d and w.p == args.p]
    if not matches:
        sys.exit(f"no QUICK_GRID workload with d={args.d}, p={args.p:g}")
    w = dataclasses.replace(matches[0], seed=SEED_TRAIN if args.seed == "train" else SEED_TEST)
    bits_list = [int(b) for b in _floats(args.bits)]
    out = args.output_dir / f"quant_sweep_{w.tag}.csv"
    meta = out.with_suffix(".json")
    if out.exists() or meta.exists():
        raise FileExistsError(f"refusing to overwrite {out}; choose a new --output-dir")
    if args.shots is not None and not 0 < args.shots <= w.shots:
        ap.error("--shots must be between 1 and the workload shot count")
    inputs = [SHOTS_DIR / f"{w.tag}.npz",
              GRAPHS_DIR / f"{w.code}_d{w.distance}_p{w.p:g}.npz"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    root = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(), *sorted((root / "src/qdec/workload").glob("*.py"))]
    }
    print(f"{w.tag} ({args.seed.upper()} shots), alpha={ALPHA}, max_iters={MAX_ITERS}, "
          f"{OSD_METHOD} order {OSD_ORDER}; range = R x max channel LLR", flush=True)
    df = sweep(w, bits_list, _floats(args.ranges), args.shots)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        df.to_csv(stream, index=False)
    provenance = measurement_info(alpha=ALPHA, max_iters=MAX_ITERS, osd_order=OSD_ORDER,
                                  osd_method=OSD_METHOD)
    provenance.update(label="FUNCTIONAL simulation; seconds are runtime, not hardware cost",
                      workload=dataclasses.asdict(w), n_shots=int(df.iloc[0]["n"]),
                      bits=bits_list, ranges=_floats(args.ranges), osd_tol=args.osd_tol,
                      input_sha256=hashes, output_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    provenance["source_sha256"] = source_hashes
    with meta.open("x") as stream:
        json.dump(provenance, stream, indent=2)
    best = pick(df, args.osd_tol)
    if best is None:
        print(f"no quantized config passes (LER and OSD rate <= {args.osd_tol:g}x float)")
    else:
        print(f"pick: quant_bits={int(best['quant_bits'])} range=x{best['range_x']:g} "
              f"(LER {best['ler']:.2e} vs float {df.iloc[0]['ler']:.2e}; OSD "
              f"{best['osd_frac']:.1%} vs {df.iloc[0]['osd_frac']:.1%})")
    print(f"wrote {out}")
    return df


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"p1_quant_sweep: missing input ({e}); run `make data` first")
        sys.exit(1)
