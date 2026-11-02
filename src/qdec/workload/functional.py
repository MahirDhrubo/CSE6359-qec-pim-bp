# OWNER: P1
"""Functional simulation: one row per SEED_TEST shot with BP behaviour and logical failures."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from qdec import io
from qdec.contracts import FUNCTIONAL_COLS, GRAPHS_DIR, SHOTS_DIR, Workload
from qdec.workload.baselines import ALPHA, MAX_ITERS, OSD_METHOD, OSD_ORDER, matching_decode_batch
from qdec.workload.bp_minsum import MinSumBP, channel_llrs
from qdec.workload.circuits import make_circuit
from qdec.workload.osd import OSD

# T11 TRAIN decision (2026-11-02): 6 bits x1.5 passes the LER screening rule and
# OSD <= 1.15x float on all four QUICK_GRID workloads. All lower-width candidates
# in the declared domain (3–8 bits, ranges x1/x1.5/x2/x3) fail somewhere.
# This is a minimum within that domain, not a proof for every possible quantizer.
# Frozen from TRAIN before the new TEST validation; see DAY2_STATUS.md.
QUANT_BITS = 6
LLR_RANGE_X = 1.5  # llr_max = this x the largest channel LLR of the workload's graph


def minsum_bposd_batch(
    H: sp.csr_matrix,
    L: sp.csr_matrix,
    priors: np.ndarray,
    dets: np.ndarray,
    obs: np.ndarray,
    *,
    quant_bits: int | None,
    llr_max: float | None,
    alpha: float,
    max_iters: int,
    osd_order: int,
    osd_method: str = "osd_cs",
    chunk: int = 128,
) -> dict[str, np.ndarray]:
    """Our min-sum BP (quantized or float) with our OSD on every non-converged shot.

    Same OSD for both quantized and float, so a quantized-vs-float LER difference comes from
    the BP messages alone. Outputs: {"fail", "bp_iters", "bp_converged", "osd_ran"} per shot.
    """
    bp = MinSumBP(H, priors, max_iters, alpha, quant_bits, llr_max=llr_max)
    osd = OSD(H, priors, osd_order, osd_method)
    dets = np.asarray(dets, dtype=bool)
    S = dets.shape[0]
    corr = np.zeros((S, H.shape[1]), dtype=np.uint8)
    iters = np.zeros(S, dtype=np.int32)
    conv = np.zeros(S, dtype=bool)
    for a in range(0, S, chunk):  # bounded memory: BP state is [chunk x n_edges]
        b = min(a + chunk, S)
        c, it, cv, llrs = bp.decode_batch(dets[a:b], return_llrs=True)
        for i in np.flatnonzero(~cv):
            c[i] = osd.decode(dets[a + i], llrs[i])
        corr[a:b], iters[a:b], conv[a:b] = c, it, cv
    predicted = (np.asarray(sp.csr_matrix(L, dtype=np.uint8) @ corr.T).T % 2).astype(bool)
    return {
        "fail": (predicted != np.asarray(obs, dtype=bool)).any(axis=1),
        "bp_iters": iters,
        "bp_converged": conv,
        "osd_ran": ~conv,
    }


def run_functional(w: Workload) -> pd.DataFrame:
    """Run quantized BP+OSD, float BP+OSD and (surface only) PyMatching on the shots of `w`.

    Inputs:  SHOTS_DIR / f"{w.tag}.npz" and the matching GRAPHS_DIR npz (via io.load_npz).
    Outputs: DataFrame matching contracts.FUNCTIONAL_COLS, one row per shot (shot_id = row index).
    Contract: written via io.write_table to TRACES_DIR / f"functional_{w.tag}.parquet".
    Owner:   P1

    bp_iters / bp_converged / osd_ran describe the quantized decoder (the system being modeled).
    fail_bp_float uses our float BP + the same OSD, so the quantized-vs-float comparison is
    paired and differs only in BP messages. Regression tests compare the float pipeline
    with ldpc on limited samples; OSD sorting ties can differ.
    """
    s = io.load_npz(SHOTS_DIR / f"{w.tag}.npz")
    g = io.load_npz(GRAPHS_DIR / f"{w.code}_d{w.distance}_p{w.p:g}.npz")
    dets, obs = s["dets"], s["obs"]
    if len(dets) != w.shots or len(obs) != w.shots:
        raise ValueError(f"{w.tag}: shot file count does not match workload ({w.shots})")
    H, L, priors = g["H"], g["L"], g["priors"]
    common = {"alpha": ALPHA, "max_iters": MAX_ITERS, "osd_order": OSD_ORDER,
              "osd_method": OSD_METHOD}
    llr_max = LLR_RANGE_X * float(channel_llrs(priors).max())
    q = minsum_bposd_batch(H, L, priors, dets, obs, quant_bits=QUANT_BITS, llr_max=llr_max,
                           **common)
    f = minsum_bposd_batch(H, L, priors, dets, obs, quant_bits=None, llr_max=None, **common)
    n = dets.shape[0]
    if w.code == "surface":
        fail_matching = pd.array(matching_decode_batch(make_circuit(w), dets, obs)["fail"],
                                 dtype="boolean")
    else:  # matching does not apply to the gross code's hypergraph
        fail_matching = pd.array([pd.NA] * n, dtype="boolean")
    df = pd.DataFrame({
        "shot_id": np.arange(n),
        "tag": w.tag,
        "n_defects": dets.sum(axis=1),
        "quant_bits": QUANT_BITS,
        "bp_iters": q["bp_iters"],
        "bp_converged": q["bp_converged"],
        "osd_ran": q["osd_ran"],
        "fail_bp_osd": q["fail"],
        "fail_bp_float": f["fail"],
        "fail_matching": fail_matching,
    })
    df = df.astype(FUNCTIONAL_COLS)
    # Diagnostic metadata only; no additions to the frozen functional table schema.
    df.attrs["decoder_settings"] = {**common, "quant_bits": QUANT_BITS,
                                    "range_x": LLR_RANGE_X, "llr_max": llr_max,
                                    "chunk": 128}
    df.attrs["float_diagnostics"] = {"osd_frac": float(f["osd_ran"].mean()),
                                     "iters_mean": float(f["bp_iters"].mean()),
                                     "iters_p99": float(np.percentile(f["bp_iters"], 99))}
    return df
