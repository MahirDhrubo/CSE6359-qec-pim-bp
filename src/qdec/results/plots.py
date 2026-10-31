# OWNER: P3
"""One function per figure. Each takes already-loaded tables and writes one file under RESULTS_DIR.

Every figure showing hardware latency/energy must say MODELED in its title or caption.
"""
from pathlib import Path

import pandas as pd


def plot_backlog_explosion(service_df: pd.DataFrame, out: Path) -> Path:
    """Queue depth over time for two systems with the same mean but different tails.
    Inputs: SERVICE_COLS (or synthetic heavy-tailed samples on Day 1). Owner: P3."""
    raise NotImplementedError


def plot_ler(functional_dfs: dict[str, pd.DataFrame], cache_dfs: dict[str, pd.DataFrame],
             out: Path) -> Path:
    """Logical error rate vs p per distance (quantized BP+OSD, float BP+OSD, matching, cache path).
    Inputs: FUNCTIONAL_COLS + CACHE_COLS per tag. Owner: P3."""
    raise NotImplementedError


def plot_latency_cdf(service_df: pd.DataFrame, out: Path) -> Path:
    """Latency CDF per System (log x), p99 / p99.9 marked. Inputs: SERVICE_COLS. Owner: P3."""
    raise NotImplementedError


def plot_backlog_over_time(sim_results: dict, out: Path) -> Path:
    """Queue depth vs time per System from queue.sim.SimResult. Owner: P3."""
    raise NotImplementedError


def plot_max_rate(rates: pd.DataFrame, out: Path) -> Path:
    """Max sustainable round rate per System and workload (from queue.sim.max_sustainable_rate).
    Owner: P3."""
    raise NotImplementedError


def plot_hit_rate(cache_dfs: dict[str, pd.DataFrame], out: Path) -> Path:
    """Cache hit rate vs p and d (and vs CAM size if provided). Inputs: CACHE_COLS. Owner: P3."""
    raise NotImplementedError


def plot_bytes_energy(service_df: pd.DataFrame, out: Path) -> Path:
    """Bytes moved and energy per decode per System (MODELED). Inputs: SERVICE_COLS. Owner: P3."""
    raise NotImplementedError


def plot_ablation(summary: pd.DataFrame, out: Path) -> Path:
    """Ablation: pim_nocache -> pim_cache -> full (p99 / p99.9 and max rate). Owner: P3."""
    raise NotImplementedError


def plot_iters_hist(functional_df: pd.DataFrame, out: Path) -> Path:
    """Histogram of bp_iters (log y), non-converged shots highlighted. Inputs: FUNCTIONAL_COLS.
    Owner: P3."""
    raise NotImplementedError


ALL_PLOTS = (
    plot_backlog_explosion, plot_ler, plot_latency_cdf, plot_backlog_over_time, plot_max_rate,
    plot_hit_rate, plot_bytes_energy, plot_ablation, plot_iters_hist,
)
