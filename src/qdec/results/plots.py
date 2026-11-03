# OWNER: P3
"""One function per figure. Each takes already-loaded tables and writes one file under RESULTS_DIR.

Every figure showing hardware latency/energy must say MODELED in its title or caption.

Robustness rules shared by every plot (real data has zeros, NaNs and missing systems):
- NaN/Inf values are dropped with a warning; a system left with no data is skipped with a
  warning instead of crashing.
- Log-axis limits come only from finite, positive values (_log_limits).
- A rate with zero observed failures is drawn as its 95% upper bound 3/N ("rule of three") with a
  hollow down-triangle marker and a legend note, never dropped (failure_rate, _plot_rates).
"""
import re
import warnings
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


def _no_minor_labels(ax, axis: str = "x") -> None:
    """Narrow log ranges make matplotlib label minor ticks, which collide; keep majors only."""
    getattr(ax, f"{axis}axis").set_minor_formatter(matplotlib.ticker.NullFormatter())


def _clean(values, what: str) -> np.ndarray:
    """Float array without NaN/Inf; warns with a count when anything was dropped."""
    v = np.asarray(values, dtype=np.float64)
    bad = ~np.isfinite(v)
    if bad.any():
        warnings.warn(f"{what}: dropped {bad.sum()} NaN/Inf of {len(v)} values", stacklevel=3)
    return v[~bad]


def _series(df: pd.DataFrame, system: str, column: str, what: str) -> np.ndarray | None:
    """One system's `column` in shot order, cleaned; None (with a warning) if nothing is left."""
    g = df[df["system"] == system]
    if "shot_id" in g:
        g = g.sort_values("shot_id")
    v = _clean(g[column].to_numpy(), f"{what}: {system}")
    if len(v) == 0:
        warnings.warn(f"{what}: skipping system {system!r}, no data", stacklevel=2)
        return None
    return v


def _log_limits(values, extra=(), pad=(0.5, 4.0), fallback=(1.0, 10.0)) -> tuple[float, float]:
    """(lo, hi) for a log axis from the finite positive entries of values + extra."""
    v = np.concatenate([np.ravel(np.asarray(values, dtype=np.float64)),
                        np.asarray([e for e in extra if e is not None], dtype=np.float64)])
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) == 0:
        return fallback
    return float(v.min() * pad[0]), float(v.max() * pad[1])


def failure_rate(k: int, n: int) -> tuple[float, bool]:
    """Observed failure rate k/n, or its 95% upper bound 3/n when k == 0 (rule of three).

    Returns (rate, is_upper_bound); (nan, False) when n == 0.
    """
    if n <= 0:
        return float("nan"), False
    if k == 0:
        return 3.0 / n, True
    return k / n, False


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k failures in n trials (P3's own copy: no cross-imports)."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def _plot_rates(ax, x, k, n, color: str, label: str) -> bool:
    """Plot failure rates with Wilson error bars; k == 0 points become hollow down-triangles at
    the 3/N upper bound. Returns True if any upper bound was drawn (caller adds the legend note).
    """
    pts = [(xi, *failure_rate(int(ki), int(ni)), int(ki), int(ni))
           for xi, ki, ni in zip(x, k, n) if ni > 0]
    if not pts:
        return False
    pts.sort()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, linewidth=1.4, label=label, zorder=2)
    seen = [p for p in pts if not p[2]]
    if seen:
        lo_hi = np.array([wilson_interval(p[3], p[4]) for p in seen])
        y = np.array([p[1] for p in seen])
        ax.errorbar([p[0] for p in seen], y, yerr=[y - lo_hi[:, 0], lo_hi[:, 1] - y], fmt="o",
                    color=color, markersize=5, capsize=2, linewidth=1, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
    bounds = [p for p in pts if p[2]]
    if bounds:
        ax.plot([p[0] for p in bounds], [p[1] for p in bounds], "v", markersize=7, zorder=3,
                markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=1.4)
    return bool(bounds)


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
        s = _series(service_df, lb, "latency_ns", "plot_backlog_explosion")
        if s is None:
            continue
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
    if ax.lines:
        _legend(ax, loc="upper left")
    return _save(fig, out)


_TAG_RE = re.compile(r"^(?P<code>[a-z]+)_d(?P<d>\d+)_p(?P<p>[^_]+)_s(?P<seed>\d+)$")
# (label, source table, failure column); the cache series only counts cache hits
LER_SERIES = (
    ("BP+OSD quantized", "functional", "fail_bp_osd"),
    ("BP+OSD float (reference)", "functional", "fail_bp_float"),
    ("PyMatching", "functional", "fail_matching"),
    ("cache path (hits only)", "cache", "fail_cache"),
)


def _counts(df: pd.DataFrame | None, column: str) -> tuple[int, int]:
    """(failures, trials) over the non-null entries of a bool/boolean column."""
    if df is None or column not in df:
        return 0, 0
    v = df[column].dropna()
    return int(v.astype(bool).sum()), len(v)


def plot_ler(
    functional_dfs: dict[str, pd.DataFrame],
    cache_dfs: dict[str, pd.DataFrame],
    out: Path,
    note: str = "",
    title: str = "Logical error rate vs physical error rate",
) -> Path:
    """Logical error rate per decode window vs p, one panel per (code, distance).

    Inputs:  {tag: FUNCTIONAL_COLS df} and {tag: CACHE_COLS df} (cache may be missing tags).
    Outputs: PNG at `out`. Observed rates carry 95% Wilson error bars; a series with 0 failures
             at some p is drawn at its 95% upper bound 3/N with a hollow down-triangle. The cache
             series is conditional on a cache hit (its N is the number of hits).
    Owner:   P3
    """
    groups: dict[tuple[str, int], list[tuple[float, str]]] = {}
    for tag in functional_dfs:
        m = _TAG_RE.match(tag)
        if m is None:
            warnings.warn(f"plot_ler: skipping tag {tag!r}, not <code>_d<d>_p<p>_s<seed>",
                          stacklevel=2)
            continue
        groups.setdefault((m["code"], int(m["d"])), []).append((float(m["p"]), tag))
    labels = [s[0] for s in LER_SERIES]
    colors = _colors(labels)
    panels = sorted(groups)
    ncols = min(2, max(1, len(panels)))
    nrows = max(1, -(-len(panels) // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows + 0.9), squeeze=False)
    any_bound, all_rates = False, []
    for ax, key in zip(axes.flat, panels):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        for label, source, col in LER_SERIES:
            pts = []
            for p, tag in sorted(groups[key]):
                df = functional_dfs[tag] if source == "functional" else cache_dfs.get(tag)
                k, n = _counts(df, col)
                if n:
                    pts.append((p, k, n))
                    all_rates.append(failure_rate(k, n)[0])
            if not pts:
                warnings.warn(f"plot_ler: {key[0]} d={key[1]}: skipping {label!r}, no data",
                              stacklevel=2)
                continue
            x, k, n = zip(*pts)
            any_bound |= _plot_rates(ax, x, k, n, colors[label], label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ps = sorted({p for p, _ in groups[key]})
        ax.set_xticks(ps, [f"{p:g}" for p in ps])         # label the simulated p values exactly
        _no_minor_labels(ax)
        ax.set_title(f"{key[0]} code, d = {key[1]}", loc="left", fontsize=9, color=INK_2)
        ax.set_xlabel("physical error rate p")
        ax.set_ylabel("logical error rate per window")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    lo, hi = _log_limits(all_rates, pad=(0.3, 3.0), fallback=(1e-6, 1.0))
    for ax in axes.flat[: len(panels)]:
        ax.set_ylim(lo, min(hi, 1.0))
    handles, names = [], []
    for ax in axes.flat[: len(panels)]:
        for h, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in names:
                handles.append(h)
                names.append(lb)
    if any_bound:
        handles.append(plt.Line2D([], [], linestyle="none", marker="v", markersize=7,
                                  markerfacecolor=SURFACE, markeredgecolor=INK_2))
        names.append("0 failures: drawn at the 95% upper bound 3/N")
    sub = "error bars: 95% Wilson interval" + (f"  |  {note}" if note else "")
    fig.suptitle(f"{title}\n{sub}", x=0.02, ha="left", fontsize=11, color=INK)
    if handles:
        leg = fig.legend(handles, names, loc="lower center", ncol=2, fontsize=8, frameon=False,
                         bbox_to_anchor=(0.5, -0.02))
        for t in leg.get_texts():
            t.set_color(INK_2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, out)


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
    shown = []
    for lb in labels:
        x = _series(service_df, lb, column, "plot_latency_cdf")
        if x is None:
            continue
        x = np.sort(x)
        n = len(x)
        ccdf = (n - np.arange(n)) / n                      # P(X >= x_i), smallest value 1/n
        p99, p999 = np.quantile(x, [0.99, 0.999])
        pos = x > 0              # the fraction uses every shot; 0 ns points just can't sit on log x
        zeros = n - int(pos.sum())
        label = f"{lb}: p99 {p99:,.0f} ns, p99.9 {p999:,.0f} ns"
        if zeros:
            label += f" ({zeros / n:.0%} of shots at 0 ns, off the log axis)"
        if pos.any():
            ax.step(x[pos], ccdf[pos], where="post", color=colors[lb], linewidth=1.4, label=label)
            shown.append(x[pos])
        else:
            warnings.warn(f"plot_latency_cdf: system {lb!r} has only 0 ns latencies", stacklevel=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*_log_limits(np.concatenate(shown) if shown else [], pad=(0.7, 1.5)))
    ax.set_ylim(top=1.05)
    for level, name in ((1e-2, "p99"), (1e-3, "p99.9")):
        ax.axhline(level, color=AXIS, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.annotate(name, (1, level), xycoords=("axes fraction", "data"), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=MUTED)
    ax.set_xlabel(f"{column.removesuffix('_ns')} (ns, log scale)")
    ax.set_ylabel("fraction of shots with latency ≥ x")
    _titles(ax, title, tag + (f"  |  {note}" if note else ""))
    if shown:
        _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    return _save(fig, out)


def plot_backlog_over_time(sim_results: dict, out: Path) -> Path:
    """Queue depth vs time per System from queue.sim.SimResult. Owner: P3."""
    raise NotImplementedError


def plot_max_rate(
    rates: pd.DataFrame,
    out: Path,
    target_hz: float | None = 1e6,
    note: str = "",
    title: str = "Max sustainable syndrome-round rate",
) -> Path:
    """Max sustainable round rate per System, one panel per workload tag.

    Inputs:  rates with columns tag, system, rate_hz (from queue.sim.max_sustainable_rate);
             target_hz = reference line (default 1 MHz = 1 us rounds, superconducting-qubit
             scale; None to omit); note = subtitle text (state the criterion and MODELED/MOCK).
    Outputs: PNG at `out`. Bars on a log x-axis, each labelled with its rate.
    Owner:   P3
    """
    tags = list(dict.fromkeys(rates["tag"]))
    ncols = 1 if len(tags) == 1 else 2
    nrows = -(-len(tags) // ncols)
    systems = list(dict.fromkeys(rates["system"]))
    colors = _colors(systems)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 0.32 * len(systems) * nrows + 1.2),
                             sharex=True, squeeze=False)
    lo, hi = _log_limits(rates["rate_hz"], extra=(target_hz,), pad=(0.5, 4.0),
                         fallback=(1e3, 1e7))
    for ax, tag in zip(axes.flat, tags):
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        d = rates[rates["tag"] == tag].set_index("system").reindex(systems)
        v_all = d["rate_hz"].to_numpy(dtype=np.float64)
        y = np.arange(len(systems))[::-1]
        drawn = np.isfinite(v_all) & (v_all > 0)
        ax.barh(y[drawn], v_all[drawn], height=0.62,
                color=[colors[s] for s, f in zip(systems, drawn) if f])
        for yi, sname, v in zip(y, systems, v_all):
            if not np.isfinite(v):
                warnings.warn(f"plot_max_rate: {tag}/{sname} has no finite rate ({v})",
                              stacklevel=2)
                text = "no data"
            elif v <= 0:
                text = "never meets the latency budget"
            else:
                text = f"{v / 1e6:.2f} MHz" if v >= 1e6 else f"{v / 1e3:.0f} kHz"
            ax.annotate(text, (v if np.isfinite(v) and v > 0 else lo, yi), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color=INK_2,
                        bbox={"fc": SURFACE, "ec": "none", "pad": 0.6})
        ax.set_yticks(y, systems)
        ax.set_ylim(-0.7, len(systems) - 0.2)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        _no_minor_labels(ax)
        if target_hz:
            ax.axvline(target_hz, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.set_title(tag, loc="left", fontsize=9, color=INK_2)
    for ax in axes.flat[len(tags):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("round rate (Hz, log scale)")
    sub = (f"dashed line: {target_hz / 1e6:g} MHz target" if target_hz else "") + (
        f"  |  {note}" if note else "")
    fig.suptitle(f"{title}\n{sub}", x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return _save(fig, out)


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
