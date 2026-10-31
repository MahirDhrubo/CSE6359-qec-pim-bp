# OWNER: P1
import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp
from ldpc import BpDecoder

from qdec.contracts import QUICK_GRID
from qdec.workload.bp_minsum import MinSumBP, channel_llrs
from qdec.workload.circuits import make_circuit, sample_shots
from qdec.workload.graphs import dem_to_graph

MAX_ITERS, ALPHA = 50, 0.625
# Test-only message range (multiple of the largest channel LLR); the real one comes from T11.
RANGE = 2.0


def _bp(H, priors, max_iters, alpha, bits):
    llr_max = None if bits is None else RANGE * channel_llrs(priors).max()
    return MinSumBP(H, priors, max_iters, alpha, bits, llr_max=llr_max)


@pytest.fixture(scope="module")
def surface():
    w = dataclasses.replace(QUICK_GRID[1], shots=300)  # d=5, p=3e-3
    return dem_to_graph(make_circuit(w)), sample_shots(w)["dets"]


def _repetition_code(n=5):
    H = sp.csr_matrix(np.eye(n - 1, n, dtype=np.uint8) + np.eye(n - 1, n, k=1, dtype=np.uint8))
    return H, np.full(n, 0.05)


def test_float_matches_ldpc_min_sum_exactly(surface):
    g, dets = surface
    ref = BpDecoder(g["H"], error_channel=list(g["priors"]), max_iter=MAX_ITERS,
                    bp_method="minimum_sum", ms_scaling_factor=ALPHA, schedule="parallel")
    corr, iters, conv = MinSumBP(g["H"], g["priors"], MAX_ITERS, ALPHA, None).decode_batch(dets)
    for i, d in enumerate(dets):
        if not d.any():  # ldpc leaves stale iter/converge on empty syndromes
            continue
        np.testing.assert_array_equal(corr[i], ref.decode(d.astype(np.uint8)))
        assert conv[i] == ref.converge
        assert iters[i] == (ref.iter if ref.converge else MAX_ITERS)


@pytest.mark.parametrize("bits", [None, 6])
def test_decode_matches_decode_batch(surface, bits):
    g, dets = surface
    bp = _bp(g["H"], g["priors"], MAX_ITERS, ALPHA, bits)
    corr, iters, conv = bp.decode_batch(dets[:40])
    for i in range(40):
        c, it, cv = bp.decode(dets[i])
        np.testing.assert_array_equal(c, corr[i])
        assert (it, cv) == (iters[i], conv[i])


@pytest.mark.parametrize("bits", [None, 4, 6, 8])
def test_iteration_and_convergence_contract(surface, bits):
    g, dets = surface
    corr, iters, conv = _bp(g["H"], g["priors"], MAX_ITERS, ALPHA, bits).decode_batch(dets)
    assert corr.dtype == np.uint8 and iters.dtype == np.int32 and conv.dtype == bool
    assert (iters[~conv] == MAX_ITERS).all() and ((iters >= 0) & (iters <= MAX_ITERS)).all()
    satisfied = ((np.asarray(g["H"] @ corr.T).T % 2).astype(bool) == dets).all(axis=1)
    assert (satisfied == conv).all()  # converged exactly when the syndrome is matched
    empty = ~dets.any(axis=1)
    assert empty.any() and (iters[empty] == 0).all() and conv[empty].all()
    assert not corr[empty].any()


def test_messages_live_on_the_fixed_point_grid():
    H, priors = _repetition_code()
    bp = MinSumBP(H, priors, 10, ALPHA, quant_bits=4, llr_max=6.0)
    assert bp.llr_max == 6.0 and bp._qmax == 7 and bp.step == pytest.approx(6.0 / 7)
    x = np.array([-100.0, -0.01, 0.0, 0.3, 1.234, 100.0])
    q = bp._q(x)
    np.testing.assert_allclose(q / bp.step, np.round(q / bp.step))  # integer multiples of step
    assert q.max() == pytest.approx(bp.llr_max) and q.min() == pytest.approx(-bp.llr_max)
    assert np.abs(bp.channel / bp.step - np.round(bp.channel / bp.step)).max() < 1e-9


def test_float_mode_does_not_quantize():
    H, priors = _repetition_code()
    bp = MinSumBP(H, priors, 10, ALPHA, None)
    x = np.array([-123.456, 0.001, 7.5])
    np.testing.assert_array_equal(bp._q(x), x)


def test_many_bits_approaches_float(surface):
    g, dets = surface
    f = MinSumBP(g["H"], g["priors"], MAX_ITERS, ALPHA, None).decode_batch(dets)
    q = _bp(g["H"], g["priors"], MAX_ITERS, ALPHA, 16).decode_batch(dets)
    assert (f[0] == q[0]).all(axis=1).mean() > 0.95


@pytest.mark.parametrize("bits", [None, 3, 4])
def test_repetition_code_single_errors(bits):
    H, priors = _repetition_code()
    errs = np.eye(5, dtype=np.uint8)
    syn = (H @ errs.T % 2).T.astype(bool)
    corr, _, conv = _bp(H, priors, 20, 1.0, bits).decode_batch(syn)
    assert conv.all()
    np.testing.assert_array_equal(corr, errs)


def test_range_at_channel_max_cannot_fix_end_errors():
    # Why llr_max is explicit: saturating at the largest channel LLR caps every check message at
    # alpha x the prior, so the single errors at both ends of the chain are never corrected.
    H, priors = _repetition_code()
    errs = np.eye(5, dtype=np.uint8)
    syn = (H @ errs.T % 2).T.astype(bool)
    bp = MinSumBP(H, priors, 20, ALPHA, 8, llr_max=channel_llrs(priors).max())
    corr, _, _ = bp.decode_batch(syn)
    assert (corr == errs).all(axis=1).tolist() == [False, True, True, True, False]


def test_channel_llrs():
    np.testing.assert_allclose(channel_llrs(np.array([0.5, 0.1])), [0.0, np.log(9)])


def test_rejects_too_few_bits_or_missing_range():
    H, priors = _repetition_code()
    with pytest.raises(ValueError):
        MinSumBP(H, priors, 10, ALPHA, quant_bits=1, llr_max=3.0)
    with pytest.raises(ValueError):
        MinSumBP(H, priors, 10, ALPHA, quant_bits=6)  # no default range
    with pytest.raises(ValueError):
        MinSumBP(H, priors, 10, ALPHA, quant_bits=6, llr_max=0.0)


def test_return_llrs_are_the_final_posteriors(surface):
    g, dets = surface
    bp = _bp(g["H"], g["priors"], MAX_ITERS, ALPHA, None)
    corr, iters, _ = bp.decode_batch(dets)
    c2, i2, _, llrs = bp.decode_batch(dets, return_llrs=True)
    np.testing.assert_array_equal(corr, c2)
    np.testing.assert_array_equal(iters, i2)
    assert llrs.shape == corr.shape
    np.testing.assert_array_equal((llrs < 0).astype(np.uint8), corr)  # hard decision of posterior
    empty = ~dets.any(axis=1)
    np.testing.assert_allclose(llrs[empty], np.broadcast_to(bp.channel, llrs[empty].shape))
