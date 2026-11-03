# OWNER: P1
"""Validate functional handoff rows and summarize paired TEST outcomes."""
import numpy as np

from qdec.contracts import SEED_TEST
from qdec.workload.baselines import MAX_ITERS
from qdec.workload.stats import cis_overlap, paired_worse_pvalue, wilson_ci


def summarize_functional(w, df, shots, quant_bits):
    """Fail closed on incomplete/misaligned files; return a diagnostic dictionary."""
    if w.seed != SEED_TEST or len(df) != w.shots or len(shots["dets"]) != w.shots:
        raise ValueError("expected complete TEST workload")
    if not df.tag.eq(w.tag).all() or not np.array_equal(df.shot_id, np.arange(w.shots)):
        raise ValueError("tag or shot_id mismatch")
    if not df.quant_bits.eq(quant_bits).all():
        raise ValueError("quantization width mismatch")
    if not np.array_equal(df.n_defects, shots["dets"].sum(axis=1)):
        raise ValueError("defect counts do not match input shots")
    if (not df.osd_ran.eq(~df.bp_converged).all()
            or not df.bp_iters.between(0, MAX_ITERS).all()
            or not df.loc[~df.bp_converged, "bp_iters"].eq(MAX_ITERS).all()):
        raise ValueError("invalid BP/OSD bookkeeping")
    if w.code == "surface" and df.fail_matching.isna().any():
        raise ValueError("surface matching results missing")
    n = len(df)
    q, f = int(df.fail_bp_osd.sum()), int(df.fail_bp_float.sum())
    qci, fci = wilson_ci(q, n), wilson_ci(f, n)
    only_q, only_f, p = paired_worse_pvalue(df.fail_bp_osd, df.fail_bp_float)
    return {"tag": w.tag, "n": n, "quant_bits": quant_bits,
            "quant_fails": q, "float_fails": f, "quant_ler": q / n, "float_ler": f / n,
            "quant_ci": qci, "float_ci": fci,
            "matching_fails": int(df.fail_matching.sum()) if w.code == "surface" else None,
            "only_q_fails": only_q, "only_float_fails": only_f, "p_worse": p,
            "ler_screen_pass": cis_overlap(qci, fci) and p >= .05,
            "quant_osd_frac": float(df.osd_ran.mean()),
            "iters_mean": float(df.bp_iters.mean()),
            "iters_p99": float(np.percentile(df.bp_iters, 99)),
            "iters_p999": float(np.percentile(df.bp_iters, 99.9))}
