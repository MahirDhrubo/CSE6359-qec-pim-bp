# OWNER: P3
"""Synthetic service times for queueing experiments that need no upstream data."""
import numpy as np
import pandas as pd


def same_mean_lognormal(
    mean_ns: float, sigmas: dict[str, float], n: int, seed: int
) -> pd.DataFrame:
    """Lognormal service times that all share one mean but differ in tail weight.

    For sigma s the log-mean is mu = ln(mean_ns) - s**2 / 2, so E[X] = mean_ns for every system.
    Heavy-tailed sample means are noisy (sigma=1.5, n=20k is off by ~5%), so each sample is then
    rescaled to have a realized mean of exactly mean_ns; the shape (sigma) is unchanged.
    Inputs:  mean_ns, {system_label: sigma}, shots per system, seed (one rng for all systems).
    Outputs: DataFrame with columns system, shot_id, latency_ns (the subset of SERVICE_COLS
             that the queue simulator and plot_backlog_explosion read).
    """
    rng = np.random.default_rng(seed)
    frames = []
    for label, sigma in sigmas.items():
        mu = np.log(mean_ns) - sigma ** 2 / 2
        x = rng.lognormal(mu, sigma, n)
        frames.append(pd.DataFrame({
            "system": label, "shot_id": np.arange(n), "latency_ns": x * (mean_ns / x.mean()),
        }))
    return pd.concat(frames, ignore_index=True)
