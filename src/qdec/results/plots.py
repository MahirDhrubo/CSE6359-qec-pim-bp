# OWNER: P3
"""One function per figure. Each takes already-loaded tables and writes one file under RESULTS_DIR.

Every figure showing hardware latency/energy must say MODELED in its title or caption.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qdec.contracts import System
from qdec.queue.sim import simulate

# Reference categorical palette (light mode), fixed order. A System always gets the same slot,
# so colors follow the entity across every figure; other labels take slots in order of appearance.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SYSTEM_COLOR = {s.value: SERIES[i] for i, s in enumerate(System)}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)


def _colors(labels) -> dict[str, str]:
    free = iter(c for c in SERIES if c not in {SYSTEM_COLOR.get(lb) for lb in labels})
    return {lb: SYSTEM_COLOR.get(lb) or next(free) for lb in labels}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _titles(ax, title: str, sub: str) -> None:
    ax.set_title(f"{title}\n", loc="left", fontsize=12, color=INK)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK_2)


def _legend(ax, **kw) -> None:
    leg = ax.legend(fontsize=8, frameon=False, **kw)
    for t in leg.get_texts():
        t.set_color(INK_2)


def _one_tag(df: pd.DataFrame) -> str:
    tags = df["tag"].unique() if "tag" in df else ["(synthetic)"]
    if len(tags) != 1:
        raise ValueError(f"expected rows for one tag, got {len(tags)}; filter by tag first")
    return str(tags[0])


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_backlog_explosion(
    service_df: pd.DataFrame,
    out: Path,
    round_period_ns: float,
    systems: list[str] | None = None,
    title: str = "Same mean service time, different tails",
    note: str = "",
) -> Path:
    """Queue depth over time per system, all fed at the same round period.

    Inputs:  service_df with columns system, shot_id, latency_ns (SERVICE_COLS rows for one tag,
             or queue.synthetic.same_mean_lognormal); systems to show (default: all, in order of
             appearance); note = extra subtitle text (e.g. "MOCK DATA", "MODELED").
    Outputs: PNG at `out`. Each series is labelled with its mean, p99.9 latency and peak depth.
    Owner:   P3
    """
    labels = systems or list(dict.fromkeys(service_df["system"]))
    colors = _colors(labels)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _style(ax)
    for lb in labels:
        s = service_df[service_df["system"] == lb].sort_values("shot_id")["latency_ns"].to_numpy()
        r = simulate(s, round_period_ns)
        p999 = np.quantile(r.latency_ns, 0.999) / round_period_ns
        ax.step(np.arange(len(s)), r.queue_depth, where="post", color=colors[lb], linewidth=1.4,
                label=f"{lb}: mean {s.mean() / round_period_ns:.2f}T, "
                      f"p99.9 latency {p999:.0f}T, peak {r.queue_depth.max()} windows"
                      f"{'' if r.stable else ' (UNSTABLE)'}")
        if len(labels) <= 4:  # beyond 4 series end labels collide; the legend carries identity
            ax.annotate(lb, (len(s) - 1, r.queue_depth[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("decode window (one per syndrome round)")
    ax.set_ylabel("queue depth (windows)")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.01)
    _titles(ax, title, f"round period T = {round_period_ns:g} ns" + (f"  |  {note}" if note else ""))
    _legend(ax, loc="upper left")
    return _save(fig, out)


def plot_ler(functional_dfs: dict[str, pd.DataFrame], cache_dfs: dict[str, pd.DataFrame],
             out: Path) -> Path:
    """Logical error rate vs p per distance (quantized BP+OSD, float BP+OSD, matching, cache path).
    Inputs: FUNCTIONAL_COLS + CACHE_COLS per tag. Owner: P3."""
    raise NotImplementedError


def plot_latency_cdf(
    service_df: pd.DataFrame,
    out: Path,
    column: str = "latency_ns",
    note: str = "",
    title: str = "Decode latency tail per system",
) -> Path:
    """Latency distribution per System drawn as the exceedance curve P(latency >= x) = 1 - CDF
    on log-log axes, so p99 and p99.9 are the horizontal levels 1e-2 and 1e-3.

    Inputs:  SERVICE_COLS rows for ONE tag (column = latency_ns, or a queueing latency column);
             note = subtitle text, e.g. "MODELED except bposd_cpu" or "MOCK DATA".
    Outputs: PNG at `out`. Legend entries carry each system's p99 and p99.9.
    Owner:   P3
    """
    tag = _one_tag(service_df)
    labels = list(dict.fromkeys(service_df["system"]))
    colors = _colors(labels)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    for lb in labels:
        x = np.sort(service_df.loc[service_df["system"] == lb, column].to_numpy())
        n = len(x)
        ccdf = (n - np.arange(n)) / n                      # P(X >= x_i), smallest value 1/n
        p99, p999 = np.quantile(x, [0.99, 0.999])
        ax.step(x, ccdf, where="post", color=colors[lb], linewidth=1.4,
                label=f"{lb}: p99 {p99:,.0f} ns, p99.9 {p999:,.0f} ns")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=1.05)
    for level, name in ((1e-2, "p99"), (1e-3, "p99.9")):
        ax.axhline(level, color=AXIS, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.annotate(name, (1, level), xycoords=("axes fraction", "data"), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=MUTED)
    ax.set_xlabel(f"{column.removesuffix('_ns')} (ns, log scale)")
    ax.set_ylabel("fraction of shots with latency ≥ x")
    _titles(ax, title, tag + (f"  |  {note}" if note else ""))
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    return _save(fig, out)


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
