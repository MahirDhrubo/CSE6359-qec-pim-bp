# OWNER: P3
"""Summary tables for the report and slides."""
import pandas as pd


def summary_table(service_df: pd.DataFrame) -> pd.DataFrame:
    """Tail-latency summary per (tag, system).

    Inputs:  SERVICE_COLS rows (COSTS_DIR/service_<tag>.parquet, possibly concatenated).
    Outputs: DataFrame with columns tag, system, p50_ns, p99_ns, p999_ns (plus mean_ns for
             reference; the story is the tails, not the mean).
    Contract: written to RESULTS_DIR as CSV / LaTeX.
    Owner:   P3
    """
    raise NotImplementedError
