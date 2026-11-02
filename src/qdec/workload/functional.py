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

# T11 provisional decision (2026-10-31, TRAIN sweeps at d5/d7 x p=1e-3/3e-3):
# fully covered setting with no significant LER degradation detected (paired McNemar)
# AND OSD rate <= 1.15x float at every point. Minimal width is not established:
# 6-bit x1.5 lacks d7 TRAIN coverage, and 7 bits has not been measured.
# 5 bits passed at p=3e-3 but ran OSD 1.38-1.56x as often as float at p=1e-3; 6 bits x1 was
# 1.21x at d5 p=1e-3. 8 bits x1: OSD 1.01-1.03x float everywhere measured.
QUANT_BITS = 8
LLR_RANGE_X = 1.0  # llr_max = this x the largest channel LLR of the workload's graph


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
    chunk: int = 2_000,
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
    paired and differs only in BP messages (float BP+OSD matches ldpc BpOsdDecoder; see tests).
    """
    s = io.load_npz(SHOTS_DIR / f"{w.tag}.npz")
    g = io.load_npz(GRAPHS_DIR / f"{w.code}_d{w.distance}_p{w.p:g}.npz")
    dets, obs = s["dets"], s["obs"]
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
    return df.astype(FUNCTIONAL_COLS)
