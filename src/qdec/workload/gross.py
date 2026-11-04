# OWNER: P1
"""Z-memory experiment for the bivariate-bicycle [[144,12,12]] code.

Construction and seven-layer schedule: Bravyi et al., arXiv:2308.07915;
authors' reference: https://github.com/sbravyi/BivariateBicycleCodes/blob/main/decoder_setup.py
A=x^3+y+y^2, B=y^3+x+x^2 on Z_12 x Z_6; HX=[A|B], HZ=[B.T|A.T].
We independently compute twelve logical Z representatives of ker(HX)/row(HZ).

Noise follows this project's surface memory convention: depolarization on data
before each round and after each CNOT; flips after reset and before measurement.
No extra per-idle-layer noise. This is not an exact reproduction of the paper's
noise experiment. Z memory measures block failure of twelve logical Zs, not
simultaneous X/Z memory. Coordinates are schematic torus labels, not planar
distances suitable for assuming surface-code cache isolation.
"""
from functools import lru_cache

import numpy as np
import stim


def rref_binary(matrix):
    """Gauss-Jordan elimination over GF(2), returning reduced matrix and pivots."""
    a = np.asarray(matrix, dtype=np.uint8).copy() % 2
    pivots = []
    for col in range(a.shape[1]):
        row = len(pivots)
        candidates = np.flatnonzero(a[row:, col])
        if not len(candidates):
            continue
        other = row + int(candidates[0])
        a[[row, other]] = a[[other, row]]
        mask = a[:, col].astype(bool)
        mask[row] = False
        a[mask] ^= a[row]
        pivots.append(col)
        if len(pivots) == a.shape[0]:
            break
    return a, pivots


def _logical_z(hx, hz):
    reduced, pivots = rref_binary(hx)
    span = {}

    def insert(v):
        bits = sum(1 << int(i) for i in np.flatnonzero(v))
        while bits:
            p = bits.bit_length() - 1
            if p not in span:
                span[p] = bits
                return True
            bits ^= span[p]
        return False

    for row in hz:
        insert(row)
    logicals = []
    for free in sorted(set(range(hx.shape[1])) - set(pivots)):
        v = np.zeros(hx.shape[1], dtype=np.uint8)
        v[free] = 1
        v[pivots] = reduced[:len(pivots), free]
        if insert(v):
            logicals.append(v)
    return np.asarray(logicals)


@lru_cache(maxsize=1)
def structure():
    """Return HX, HZ, logical Zs and ordered neighbor lists (read-only arrays)."""
    ell, m = 12, 6
    n = ell * m

    def shift(dx, dy):
        i = np.arange(n)
        out = np.zeros((n, n), dtype=np.uint8)
        out[i, ((i // m + dx) % ell) * m + (i % m + dy) % m] = 1
        return out

    aa = [shift(3, 0), shift(0, 1), shift(0, 2)]
    bb = [shift(0, 3), shift(1, 0), shift(2, 0)]
    a, b = sum(aa) % 2, sum(bb) % 2
    hx, hz = np.hstack([a, b]), np.hstack([b.T, a.T])
    nx = np.column_stack([x.argmax(axis=1) for x in aa]
                         + [n + x.argmax(axis=1) for x in bb])
    nz = np.column_stack([x.argmax(axis=0) for x in bb]
                         + [n + x.argmax(axis=0) for x in aa])
    logicals = _logical_z(hx, hz)
    assert not (hx @ hz.T % 2).any() and logicals.shape == (12, 144)
    for arr in (hx, hz, logicals, nx, nz):
        arr.setflags(write=False)
    return hx, hz, logicals, nx, nz


def make_gross_circuit(w):
    if w.distance != 12 or w.rounds < 1 or not 0 <= w.p <= 1:
        raise ValueError("gross requires distance=12, rounds>=1 and 0<=p<=1")
    _, hz, logicals, nx, nz = structure()
    h, n = 72, 144
    data = list(range(n))
    xs, zs = list(range(n, n + h)), list(range(n + h, n + 2 * h))
    circuit = stim.Circuit()
    for q in range(288):
        i = q % h
        circuit.append("QUBIT_COORDS", [q], [i // 6, i % 6, q // h])
    circuit.append("R", data)
    circuit.append("X_ERROR", data, w.p)
    x_order = (None, 1, 4, 3, 5, 0, 2)
    z_order = (3, 5, 0, 1, 2, 4, None)
    for t in range(w.rounds):
        circuit.append("DEPOLARIZE1", data, w.p)
        circuit.append("RX", xs)
        circuit.append("Z_ERROR", xs, w.p)
        circuit.append("R", zs)
        circuit.append("X_ERROR", zs, w.p)
        for dx, dz in zip(x_order, z_order, strict=True):
            pairs = []
            if dx is not None:
                pairs.extend(q for i in range(h) for q in (xs[i], int(nx[i, dx])))
            if dz is not None:
                pairs.extend(q for i in range(h) for q in (int(nz[i, dz]), zs[i]))
            assert len(pairs) == len(set(pairs)), "CNOT layer has a qubit collision"
            circuit.append("CX", pairs)
            circuit.append("DEPOLARIZE2", pairs, w.p)
            circuit.append("TICK")
        circuit.append("Z_ERROR", xs, w.p)
        circuit.append("MX", xs)
        circuit.append("X_ERROR", zs, w.p)
        circuit.append("M", zs)
        for i in range(h):
            if t:
                circuit.append("DETECTOR", [stim.target_rec(-2*h+i),
                                             stim.target_rec(-4*h+i)], [2*(i//6), i%6, t])
            recs = [stim.target_rec(-h+i)]
            if t:
                recs.append(stim.target_rec(-3*h+i))
            circuit.append("DETECTOR", recs, [2*(i//6)+1, i%6, t])
    circuit.append("X_ERROR", data, w.p)
    circuit.append("M", data)
    for i, check in enumerate(hz):
        recs = [stim.target_rec(-n + int(j)) for j in np.flatnonzero(check)]
        recs.append(stim.target_rec(-n-h+i))
        circuit.append("DETECTOR", recs, [2*(i//6)+1, i%6, w.rounds])
    for i, logical in enumerate(logicals):
        circuit.append("OBSERVABLE_INCLUDE",
                       [stim.target_rec(-n + int(j)) for j in np.flatnonzero(logical)], i)
    return circuit
