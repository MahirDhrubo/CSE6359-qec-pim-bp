# OWNER: P1
"""Ordered-statistics decoding (OSD-0 / OSD-CS) on externally supplied BP soft outputs.

ldpc only runs OSD inside BpOsdDecoder, after its own (float) BP. To get quantized BP+OSD we
need OSD after *our* BP, so this re-implements ldpc's src_cpp/osd.hpp:
  * columns ordered by ascending posterior LLR (least reliable first);
  * greedy GF(2) elimination in that order picks the pivot (basis) columns;
  * OSD-0: free columns = 0, pivot bits solve the syndrome;
  * OSD-CS(order w): also every weight-1 pattern on the free columns and every weight-2
    pattern on the first w free columns; keep the candidate with the lowest
    cost = sum of -log(prior) over flipped bits, replacing only on strict improvement.
One deliberate difference: ldpc sorts with an unstable qsort; we use a stable sort (ties by
column index) so results are reproducible. Ties are common (untouched variables keep their
channel LLR), so corrections can differ from ldpc on tied shots while LER stays the same.
"""
import numpy as np
import scipy.sparse as sp

OSD_METHODS = ("osd_0", "osd_cs")


class OSD:
    """OSD post-processor for one parity-check matrix.

    Inputs:  H csr [n_checks x n_vars], priors float64[n_vars] (for the candidate cost),
             osd_order (w), method "osd_cs" | "osd_0".
    decode(syndrome, llrs) -> correction uint8[n_vars] with H @ correction == syndrome.
    """

    def __init__(self, H: sp.csr_matrix, priors: np.ndarray, osd_order: int,
                 method: str = "osd_cs"):
        if method not in OSD_METHODS:
            raise ValueError(f"method must be one of {OSD_METHODS}")
        H = sp.csr_matrix(H, dtype=np.uint8)
        self.n_checks, self.n_vars = H.shape
        self.method = method
        self.osd_order = int(osd_order) if method == "osd_cs" else 0
        p = np.clip(np.asarray(priors, dtype=np.float64), 1e-300, 1.0)
        self.cost = -np.log(p)
        self.H_cols = H.tocsc()  # column access for permuting

    def decode(self, syndrome: np.ndarray, llrs: np.ndarray) -> np.ndarray:
        order = np.argsort(np.asarray(llrs, dtype=np.float64), kind="stable")
        dense = self.H_cols[:, order].toarray().astype(bool)
        R, s, pivots = _eliminate(dense, np.asarray(syndrome, dtype=bool))
        r = len(pivots)
        if s[r:].any():
            raise ValueError("syndrome is not in the column space of H")
        s = s[:r]
        R = R[:r]
        is_piv = np.zeros(self.n_vars, dtype=bool)
        is_piv[pivots] = True
        free = np.flatnonzero(~is_piv)  # positions in `order`, still in reliability order
        piv = np.asarray(pivots, dtype=np.int64)
        cost = self.cost[order]
        w_piv, w_free = cost[piv], cost[free]

        best_cost = float(w_piv @ s)
        best_basis, best_free = s, None

        if self.method == "osd_cs" and free.size:
            Rf = R[:, free]  # [r x k]
            # weight-1 on every free column
            B1 = s[:, None] ^ Rf
            c1 = w_piv @ B1 + w_free
            # weight-2 on the first `osd_order` free columns, pairs (i, j>i) in ldpc's order
            m = min(self.osd_order, free.size)
            ii, jj = np.triu_indices(m, k=1)
            B2 = s[:, None] ^ Rf[:, ii] ^ Rf[:, jj]
            c2 = w_piv @ B2 + w_free[ii] + w_free[jj]
            costs = np.concatenate([c1, c2])
            j = int(np.argmin(costs))  # first minimum = earliest candidate, like ldpc
            if costs[j] < best_cost:
                best_cost = float(costs[j])
                if j < free.size:
                    best_basis, best_free = B1[:, j], [free[j]]
                else:
                    q = j - free.size
                    best_basis, best_free = B2[:, q], [free[ii[q]], free[jj[q]]]

        e_perm = np.zeros(self.n_vars, dtype=np.uint8)
        e_perm[piv] = best_basis
        if best_free is not None:
            e_perm[best_free] = 1
        e = np.zeros(self.n_vars, dtype=np.uint8)
        e[order] = e_perm
        return e


def _eliminate(M: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Gauss-Jordan over GF(2), greedy pivots scanning columns left to right.

    Rows are bit-packed (64 columns per word) so each pivot step is one vectorized XOR.
    Returns (reduced bool matrix, transformed syndrome, pivot column positions); the first
    len(pivots) rows hold the identity on the pivot columns.
    """
    m, n = M.shape
    packed = _pack(M)
    s = s.copy()
    pivots: list[int] = []
    r = 0
    for c in range(n):
        if r == m:
            break
        word, bit = c >> 6, np.uint64(c & 63)
        col = ((packed[:, word] >> bit) & np.uint64(1)).astype(bool)
        cand = np.flatnonzero(col[r:])
        if cand.size == 0:
            continue
        i = r + int(cand[0])
        if i != r:
            packed[[r, i]] = packed[[i, r]]
            s[[r, i]] = s[[i, r]]
            col[[r, i]] = col[[i, r]]
        col[r] = False
        packed[col] ^= packed[r]
        s[col] ^= s[r]
        pivots.append(c)
        r += 1
    return _unpack(packed, n), s, pivots


def _pack(M: np.ndarray) -> np.ndarray:
    m, n = M.shape
    words = (n + 63) // 64
    padded = np.zeros((m, words * 64), dtype=np.uint8)
    padded[:, :n] = M
    return np.packbits(padded, axis=1, bitorder="little").view(np.uint64).copy()


def _unpack(P: np.ndarray, n: int) -> np.ndarray:
    return np.unpackbits(P.view(np.uint8), axis=1, bitorder="little")[:, :n].astype(bool)
