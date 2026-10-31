"""
qdec/mocks.py — fake data in the exact contract schemas, so nobody waits on anybody.

P2 develops the cost model against mock_functional(); P3 develops the queue sim and
every plot against mock_service(). When the real parquet files land, swap the loader
and nothing else changes. Delete call sites (not this file) on Day 3.
"""
import numpy as np
import pandas as pd

from qdec.contracts import CACHE_COLS, FUNCTIONAL_COLS, SERVICE_COLS, System


def mock_functional(tag: str, n: int = 20_000, max_iters: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    iters = np.minimum(rng.geometric(0.25, n), max_iters)          # heavy-ish tail
    iters[rng.random(n) < 0.01] = max_iters                        # ~1% non-converged
    conv = iters < max_iters
    df = pd.DataFrame({
        "shot_id": np.arange(n), "tag": tag,
        "n_defects": rng.poisson(4, n), "quant_bits": 6,
        "bp_iters": iters, "bp_converged": conv, "osd_ran": ~conv,
        "fail_bp_osd": rng.random(n) < 1e-3, "fail_bp_float": rng.random(n) < 1e-3,
        "fail_matching": pd.array([None] * n, dtype="boolean"),
    })
    return df.astype(FUNCTIONAL_COLS)


def mock_cache(tag: str, n: int = 20_000, hit_rate: float = 0.7, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hit = rng.random(n) < hit_rate
    fail = pd.array(np.where(hit, rng.random(n) < 1e-3, False), dtype="boolean")
    fail[~hit] = pd.NA
    df = pd.DataFrame({
        "shot_id": np.arange(n), "tag": tag,
        "n_clusters": rng.poisson(2, n), "max_cluster_size": rng.integers(1, 8, n),
        "n_lookups": rng.poisson(2, n), "cache_hit": hit, "fail_cache": fail,
    })
    return df.astype(CACHE_COLS)


def mock_service(tag: str, n: int = 20_000, seed: int = 2) -> pd.DataFrame:
    """Heavy-tailed service times: same-ish mean, very different tails per system."""
    rng = np.random.default_rng(seed)
    scale = {System.BPOSD_CPU: 900, System.BP_CPU_CONV: 600, System.NEARMEM_DPU: 400,
             System.RELAY_FPGA: 300, System.PIM_NOCACHE: 250, System.PIM_CACHE: 120, System.FULL: 120}
    rows = []
    for sys, s in scale.items():
        lat = rng.lognormal(np.log(s), 0.9 if sys == System.BPOSD_CPU else 0.5, n)
        rows.append(pd.DataFrame({
            "shot_id": np.arange(n), "tag": tag, "system": sys.value,
            "path": np.where(rng.random(n) < 0.7, "cache", "bp") if "cache" in sys.value or sys == System.FULL else "bp",
            "latency_ns": lat, "energy_nj": lat * 0.05, "bytes_moved": lat * 10,
        }))
    return pd.concat(rows, ignore_index=True).astype(SERVICE_COLS)
