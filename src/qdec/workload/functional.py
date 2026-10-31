# OWNER: P1
"""Functional simulation: one row per SEED_TEST shot with BP behaviour and logical failures."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from qdec.contracts import FUNCTIONAL_COLS, Workload  # noqa: F401
from qdec.workload.bp_minsum import MinSumBP
from qdec.workload.osd import OSD


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
    """
    raise NotImplementedError
