# OWNER: P1
"""Small statistics helpers for LER comparisons (binomial confidence intervals)."""
import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k failures in n shots (95% for z=1.96).

    Unlike the normal approximation it stays inside [0, 1] and is non-degenerate at k = 0,
    which matters at low p where a run may see zero logical failures.
    """
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)  # exact ends; avoid 1 - 1e-16 rounding
    hi = 1.0 if k == n else min(1.0, centre + half)
    return (float(lo), float(hi))


def cis_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True when two intervals share any point (a conservative 'not distinguishable' test)."""
    return a[0] <= b[1] and b[0] <= a[1]


def latency_percentiles(latency_ns: np.ndarray, qs=(50, 90, 99, 99.9)) -> dict[str, float]:
    """Per-shot latency tail summary in microseconds, plus the max (tails, not means)."""
    us = np.asarray(latency_ns, dtype=np.float64) / 1e3
    out = {f"p{q:g}": float(np.percentile(us, q)) for q in qs}
    out["max"] = float(us.max())
    return out
