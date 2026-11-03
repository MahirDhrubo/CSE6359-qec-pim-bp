# OWNER: P1
import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp
from ldpc import BpOsdDecoder

from qdec.contracts import QUICK_GRID
from qdec.workload.bp_minsum import MinSumBP
from qdec.workload.circuits import make_circuit, sample_shots
from qdec.workload.functional import minsum_bposd_batch
from qdec.workload.graphs import dem_to_graph
from qdec.workload.osd import OSD, _eliminate

MAX_ITERS, ALPHA, ORDER = 50, 0.625, 7


@pytest.fixture(scope="module")
def surface():
    w = dataclasses.replace(QUICK_GRID[1], shots=300)  # d=5, p=3e-3: ~40% need OSD
    s = sample_shots(w)
    return dem_to_graph(make_circuit(w)), s["dets"], s["obs"]


def test_eliminate_gives_identity_on_pivots_and_solves():
    rng = np.random.default_rng(3)
    M = rng.random((12, 40)) < 0.3
    x = (rng.random(40) < 0.2).astype(np.uint8)
    s = (M.astype(np.uint8) @ x % 2).astype(bool)
    R, t, piv = _eliminate(M, s)
    r = len(piv)
    np.testing.assert_array_equal(R[:r][:, piv], np.eye(r, dtype=bool))
    assert not R[r:].any() and not t[r:].any()
    assert piv == sorted(piv)  # greedy left-to-right
    sol = np.zeros(40, np.uint8)
    sol[piv] = t[:r]
    np.testing.assert_array_equal(M.astype(np.uint8) @ sol % 2, s)


def test_eliminate_across_word_boundaries():
    M = np.zeros((3, 200), dtype=bool)
    M[0, 63], M[1, 64], M[2, 199] = True, True, True
    M[0, 199] = True
    _, _, piv = _eliminate(M, np.zeros(3, bool))
    assert piv == [63, 64, 199]


def test_osd_always_satisfies_syndrome(surface):
    g, dets, _ = surface
    _, _, conv, llrs = MinSumBP(g["H"], g["priors"], MAX_ITERS, ALPHA, None).decode_batch(
        dets, return_llrs=True)
    for method in ("osd_0", "osd_cs"):
        osd = OSD(g["H"], g["priors"], ORDER, method)
        for i in np.flatnonzero(~conv)[:30]:
            e = osd.decode(dets[i], llrs[i])
            np.testing.assert_array_equal(np.asarray(g["H"] @ e) % 2, dets[i])


def test_osd_cs_no_worse_than_osd_0(surface):
    g, dets, _ = surface
    _, _, conv, llrs = MinSumBP(g["H"], g["priors"], MAX_ITERS, ALPHA, None).decode_batch(
        dets, return_llrs=True)
    cost = -np.log(g["priors"])
    o0, cs = OSD(g["H"], g["priors"], 0, "osd_0"), OSD(g["H"], g["priors"], ORDER)
    for i in np.flatnonzero(~conv)[:30]:
        assert cs.decode(dets[i], llrs[i]) @ cost <= o0.decode(dets[i], llrs[i]) @ cost + 1e-9


def test_float_pipeline_matches_ldpc_bposd(surface):
    # Our float BP + our OSD vs ldpc BpOsdDecoder: same answer except on sort ties, where ours
    # is never costlier; and the same logical failures.
    g, dets, obs = surface
    ref = BpOsdDecoder(g["H"], error_channel=list(g["priors"]), max_iter=MAX_ITERS,
                       bp_method="minimum_sum", ms_scaling_factor=ALPHA, osd_method="osd_cs",
                       osd_order=ORDER)
    rc = np.array([ref.decode(d.astype(np.uint8)) for d in dets])
    corr, _, conv, llrs = MinSumBP(g["H"], g["priors"], MAX_ITERS, ALPHA, None).decode_batch(
        dets, return_llrs=True)
    osd = OSD(g["H"], g["priors"], ORDER)
    for i in np.flatnonzero(~conv):
        corr[i] = osd.decode(dets[i], llrs[i])
    cost = -np.log(g["priors"])
    assert (corr == rc).all(axis=1).mean() > 0.97
    assert (corr @ cost <= rc @ cost + 1e-9).all()

    def n_fail(c):
        return int(((np.asarray(g["L"] @ c.T).T % 2).astype(bool) != obs).any(axis=1).sum())

    assert n_fail(corr) == n_fail(rc)


def test_minsum_bposd_batch_contract(surface):
    g, dets, obs = surface
    r = minsum_bposd_batch(g["H"], g["L"], g["priors"], dets, obs, quant_bits=6,
                           llr_max=2 * 12.0, alpha=ALPHA, max_iters=MAX_ITERS, osd_order=ORDER,
                           chunk=128)  # small chunk exercises the chunk loop
    assert set(r) == {"fail", "bp_iters", "bp_converged", "osd_ran"}
    assert all(len(v) == len(dets) for v in r.values())
    assert (r["osd_ran"] == ~r["bp_converged"]).all()
    assert (r["bp_iters"][~r["bp_converged"]] == MAX_ITERS).all()
    assert r["fail"].mean() < 0.05


def test_osd_rejects_unknown_method():
    with pytest.raises(ValueError):
        OSD(sp.csr_matrix(np.eye(2, dtype=np.uint8)), np.full(2, 0.1), 0, "osd_e")


@pytest.mark.parametrize("bits", [None, 6])
def test_chunk_size_preserves_per_shot_results(surface, bits):
    g, dets, obs = surface
    kwargs = {"quant_bits": bits, "llr_max": 18.0 if bits else None,
              "alpha": ALPHA, "max_iters": MAX_ITERS, "osd_order": ORDER}
    a = minsum_bposd_batch(g["H"], g["L"], g["priors"], dets[:40], obs[:40],
                           chunk=7, **kwargs)
    b = minsum_bposd_batch(g["H"], g["L"], g["priors"], dets[:40], obs[:40],
                           chunk=128, **kwargs)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
