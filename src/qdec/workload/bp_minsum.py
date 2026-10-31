# OWNER: P1
"""Quantized normalized min-sum belief propagation (the algorithm modeled as running in SRAM)."""
import numpy as np
import scipy.sparse as sp


def channel_llrs(priors: np.ndarray) -> np.ndarray:
    """Initial LLRs log((1-p)/p); T11 sweeps llr_max as multiples of their maximum."""
    p = np.clip(np.asarray(priors, dtype=np.float64), 1e-15, 1 - 1e-15)
    return np.log((1 - p) / p)


class MinSumBP:
    """Normalized min-sum BP with fixed-point message quantization.

    Inputs:  H          csr [n_checks x n_vars] from graphs.dem_to_graph
             priors     float64[n_vars] fault probabilities (initial LLRs = log((1-p)/p))
             max_iters  iteration cap (when not converged, bp_iters == max_iters)
             alpha      min-sum normalization factor (e.g. 0.625 - 1.0)
             quant_bits message width in bits; None = unquantized float reference
             llr_max    (keyword, required when quant_bits is set) saturation level of messages
    Contract: results feed FUNCTIONAL_COLS (bp_iters, bp_converged, quant_bits) via functional.py.
    Owner:   P1

    Schedule: flooding (parallel), like ldpc's default, so the float version can be checked
    against ldpc.BpDecoder(bp_method="minimum_sum", schedule="parallel").

    Quantization (quant_bits = b): every stored message (channel LLRs, variable-to-check and
    check-to-variable) is a signed b-bit integer times step = llr_max / (2**(b-1) - 1), i.e.
    rounded to that grid and saturated at +-llr_max. There is deliberately no default: the range
    trades resolution against saturation and is chosen per workload by the T11 sweep. It must
    exceed the largest channel LLR (by more than 1/alpha), or a check message can never
    outweigh a saturated prior and BP misses e.g. single errors at the end of a chain.
    """

    def __init__(
        self,
        H: sp.csr_matrix,
        priors: np.ndarray,
        max_iters: int,
        alpha: float,
        quant_bits: int | None,
        *,
        llr_max: float | None = None,
    ):
        H = sp.csr_matrix(H, dtype=np.uint8)
        H.sort_indices()
        H.eliminate_zeros()
        if quant_bits is not None and quant_bits < 2:
            raise ValueError("quant_bits must be >= 2 (one sign bit plus magnitude)")
        if quant_bits is not None and (llr_max is None or llr_max <= 0):
            raise ValueError("quantized BP needs an explicit llr_max > 0 (chosen by the T11 sweep)")
        self.H = H
        self.n_checks, self.n_vars = H.shape
        self.max_iters = int(max_iters)
        self.alpha = float(alpha)
        self.quant_bits = quant_bits

        # Edge lists in CSR (check-major) order; reduceat over check segments needs the starts.
        deg = np.diff(H.indptr)
        if (deg == 0).any():
            # empty checks make reduceat segments ill-defined
            raise ValueError("H has checks with no variables")
        self.edge_check = np.repeat(np.arange(self.n_checks), deg)
        self.edge_var = H.indices.astype(np.int64)
        self.check_start = H.indptr[:-1]
        n_edges = self.edge_var.size
        self.edge_id = np.arange(n_edges)
        # [E x n_vars] incidence: (edge values) @ edge_to_var sums them into their variables
        self.edge_to_var = sp.csr_matrix(
            (np.ones(n_edges), (self.edge_id, self.edge_var)), shape=(n_edges, self.n_vars)
        )

        llr = channel_llrs(priors)
        if quant_bits is None:
            self.step = None
            self.llr_max = np.inf
        else:
            self.llr_max = float(llr_max)
            self.step = self.llr_max / (2 ** (quant_bits - 1) - 1)
        self.channel = self._q(llr)

    def _q(self, x: np.ndarray) -> np.ndarray:
        """Round to the fixed-point grid and saturate (identity in float mode)."""
        if self.step is None:
            return x
        return np.clip(np.round(x / self.step), -self._qmax, self._qmax) * self.step

    @property
    def _qmax(self) -> int:
        return 2 ** (self.quant_bits - 1) - 1

    def decode(self, syndrome: np.ndarray) -> tuple[np.ndarray, int, bool]:
        """Decode one syndrome.

        Inputs:  syndrome bool[n_checks].
        Outputs: (correction uint8[n_vars], iters run, converged: H @ correction == syndrome).
        Owner:   P1
        """
        corr, iters, conv = self.decode_batch(np.asarray(syndrome, dtype=bool)[None, :])
        return corr[0], int(iters[0]), bool(conv[0])

    def decode_batch(self, syndromes: np.ndarray, return_llrs: bool = False) -> tuple:
        """Decode many syndromes at once (same result as decode() per row).

        Inputs:  syndromes bool[B x n_checks].
        Outputs: corrections uint8[B x n_vars], iters int32[B], converged bool[B]
                 (+ posterior LLRs float64[B x n_vars] at the stopping iteration if return_llrs;
                 these are the soft values OSD orders columns by).
        Each row stops independently as soon as its hard decision matches its syndrome; an
        empty syndrome is converged after 0 iterations.
        """
        syn = np.asarray(syndromes, dtype=bool)
        B = syn.shape[0]
        corr = np.zeros((B, self.n_vars), dtype=np.uint8)
        iters = np.zeros(B, dtype=np.int32)
        conv = ~syn.any(axis=1)
        llrs = np.tile(self.channel, (B, 1)) if return_llrs else None
        active = np.flatnonzero(~conv)
        if active.size == 0 or self.max_iters == 0:
            iters[active] = self.max_iters
            return (corr, iters, conv, llrs) if return_llrs else (corr, iters, conv)

        ev, ec, starts = self.edge_var, self.edge_check, self.check_start
        # check-node sign from the syndrome: a fired check wants odd parity -> flip sign
        syn_sign = np.where(syn[active][:, ec], -1.0, 1.0)
        c2v = np.zeros((active.size, ev.size))
        for it in range(1, self.max_iters + 1):
            # variable -> check: channel + all incoming except the one on this edge
            total = self._var_sums(c2v)
            v2c = self._q(self.channel[ev] + total[:, ev] - c2v)

            # check -> variable: alpha * (product of other signs) * (min of other magnitudes)
            mag = np.abs(v2c)
            neg = v2c < 0
            min1 = np.minimum.reduceat(mag, starts, axis=1)
            # the (first) edge achieving min1 in each check gets min2 instead
            at_min = np.where(mag == min1[:, ec], self.edge_id, ev.size)
            first = self.edge_id == np.minimum.reduceat(at_min, starts, axis=1)[:, ec]
            min2 = np.minimum.reduceat(np.where(first, np.inf, mag), starts, axis=1)
            other_min = np.where(first, min2[:, ec], min1[:, ec])
            parity = np.add.reduceat(neg.astype(np.int64), starts, axis=1) % 2
            sign = np.where((parity[:, ec] + neg) % 2 == 1, -1.0, 1.0) * syn_sign
            other_min = np.where(np.isinf(other_min), 0.0, other_min)  # degree-1 check
            c2v = self._q(self.alpha * sign * other_min)

            # hard decision and per-row early stop
            post = self.channel + self._var_sums(c2v)
            hard = (post < 0).astype(np.uint8)
            parity_ok = (np.asarray(self.H @ hard.T).T % 2).astype(bool) == syn[active]
            done = parity_ok.all(axis=1)
            rows = active[done]
            corr[rows] = hard[done]
            iters[rows] = it
            conv[rows] = True
            if return_llrs:
                llrs[rows] = post[done]
            if it == self.max_iters:
                rest = active[~done]
                corr[rest] = hard[~done]
                iters[rest] = self.max_iters
                if return_llrs:
                    llrs[rest] = post[~done]
                break
            keep = ~done
            if not keep.any():
                break
            active, c2v, syn_sign = active[keep], c2v[keep], syn_sign[keep]
        return (corr, iters, conv, llrs) if return_llrs else (corr, iters, conv)

    def _var_sums(self, edge_vals: np.ndarray) -> np.ndarray:
        """Sum edge values into their variables: [R x E] -> [R x n_vars]."""
        return np.asarray(edge_vals @ self.edge_to_var)
