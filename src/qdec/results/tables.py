# OWNER: P3
"""Summary tables for the report and slides."""
import pandas as pd

QUANTILES = {"p50_ns": 0.50, "p99_ns": 0.99, "p999_ns": 0.999}


def summary_table(service_df: pd.DataFrame, column: str = "latency_ns") -> pd.DataFrame:
    """Tail-latency summary per (tag, system).

    Inputs:  SERVICE_COLS rows (COSTS_DIR/service_<tag>.parquet, possibly concatenated);
             column = which latency column to summarize (default the service latency).
    Outputs: DataFrame with columns tag, system, n, mean_ns, p50_ns, p99_ns, p999_ns, max_ns
             (mean for reference only; the story is the tails, not the mean). Systems keep the
             order of their first appearance, tags are sorted.
    Contract: written to RESULTS_DIR as CSV / LaTeX.
    Owner:   P3
    """
    order = list(dict.fromkeys(service_df["system"]))
    g = service_df.groupby(["tag", "system"], observed=True, sort=False)[column]
    out = g.agg(n="size", mean_ns="mean", max_ns="max")
    for name, q in QUANTILES.items():
        out[name] = g.quantile(q)
    out = out.reset_index()[["tag", "system", "n", "mean_ns", *QUANTILES, "max_ns"]]
    out["system"] = pd.Categorical(out["system"], categories=order, ordered=True)
    out = out.sort_values(["tag", "system"], ignore_index=True)
    out["system"] = out["system"].astype("string")
    out["tag"] = out["tag"].astype("string")
    return out
