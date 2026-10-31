# OWNER: P1
"""Quantized normalized min-sum belief propagation (the algorithm modeled as running in SRAM)."""
import numpy as np
import scipy.sparse as sp


class MinSumBP:
    """Normalized min-sum BP with fixed-point message quantization.

    Inputs:  H          csr [n_checks x n_vars] from graphs.dem_to_graph
             priors     float64[n_vars] fault probabilities (initial LLRs = log((1-p)/p))
             max_iters  iteration cap (when not converged, bp_iters == max_iters)
             alpha      min-sum normalization factor (e.g. 0.625 - 1.0)
             quant_bits message width in bits; None = unquantized float reference
    Contract: results feed FUNCTIONAL_COLS (bp_iters, bp_converged, quant_bits) via functional.py.
    Owner:   P1
    """

    def __init__(
        self,
        H: sp.csr_matrix,
        priors: np.ndarray,
        max_iters: int,
        alpha: float,
        quant_bits: int | None,
    ):
        raise NotImplementedError

    def decode(self, syndrome: np.ndarray) -> tuple[np.ndarray, int, bool]:
        """Decode one syndrome.

        Inputs:  syndrome bool[n_checks].
        Outputs: (correction uint8[n_vars], iters run, converged: H @ correction == syndrome).
        Owner:   P1
        """
        raise NotImplementedError
