# OWNER: P1
"""Small statistics helpers for LER comparisons (binomial confidence intervals)."""
import numpy as np
from scipy.stats import binomtest


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


def paired_worse_pvalue(fail_a: np.ndarray, fail_b: np.ndarray) -> tuple[int, int, float]:
    """One-sided exact McNemar test that decoder A fails more often than B on the same shots.

    Only discordant shots carry information: n_ab = A fails & B ok, n_ba = B fails & A ok.
    Under "A no worse than B" each discordant shot is a fair coin, so
    p = P(X >= n_ab), X ~ Binomial(n_ab + n_ba, 1/2). Small p => A is significantly worse.
    Far more sensitive than comparing two independent CIs when the decoders mostly agree.
    """
    a, b = np.asarray(fail_a, dtype=bool), np.asarray(fail_b, dtype=bool)
    n_ab, n_ba = int((a & ~b).sum()), int((b & ~a).sum())
    n = n_ab + n_ba
    if n == 0:
        return n_ab, n_ba, 1.0
    return n_ab, n_ba, float(binomtest(n_ab, n, 0.5, alternative="greater").pvalue)


def latency_percentiles(latency_ns: np.ndarray, qs=(50, 90, 99, 99.9)) -> dict[str, float]:
    """Per-shot latency tail summary in microseconds, plus the max (tails, not means)."""
    us = np.asarray(latency_ns, dtype=np.float64) / 1e3
    out = {f"p{q:g}": float(np.percentile(us, q)) for q in qs}
    out["max"] = float(us.max())
    return out
