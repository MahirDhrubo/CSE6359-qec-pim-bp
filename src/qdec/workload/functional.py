# OWNER: P1
"""Functional simulation: one row per SEED_TEST shot with BP behaviour and logical failures."""
import pandas as pd

from qdec.contracts import FUNCTIONAL_COLS, Workload  # noqa: F401


def run_functional(w: Workload) -> pd.DataFrame:
    """Run quantized BP+OSD, float BP+OSD and (surface only) PyMatching on the shots of `w`.

    Inputs:  SHOTS_DIR / f"{w.tag}.npz" and the matching GRAPHS_DIR npz (via io.load_npz).
    Outputs: DataFrame matching contracts.FUNCTIONAL_COLS, one row per shot (shot_id = row index).
    Contract: written via io.write_table to TRACES_DIR / f"functional_{w.tag}.parquet".
    Owner:   P1
    """
    raise NotImplementedError
