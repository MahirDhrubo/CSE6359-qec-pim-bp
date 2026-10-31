# OWNER: SHARED (frozen after setup; changes go through a PR to main like contracts.py)
"""
qdec/io.py — the only way tables and arrays cross branch boundaries.

    write_table(df, FUNCTIONAL_COLS, TRACES_DIR / f"functional_{w.tag}.parquet")
    df = read_table(TRACES_DIR / f"functional_{w.tag}.parquet", FUNCTIONAL_COLS)

Both validate that the column set matches the schema exactly and cast with df.astype(cols),
so a schema violation fails loudly at the writer, not three steps downstream.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from qdec.contracts import COSTS_DIR, GRAPHS_DIR, RESULTS_DIR, SHOTS_DIR, TRACES_DIR

_SPARSE_TAG = "__csr__"


def _validate(df: pd.DataFrame, cols: dict[str, str], where: str | Path) -> pd.DataFrame:
    got, want = set(df.columns), set(cols)
    if got != want:
        raise ValueError(
            f"{where}: column mismatch. missing={sorted(want - got)} extra={sorted(got - want)}"
        )
    return df[list(cols)].astype(cols)


def write_table(df: pd.DataFrame, cols: dict[str, str], path: str | Path) -> Path:
    """Validate df against schema `cols` and write it to parquet at `path` (parents created)."""
    path = Path(path)
    out = _validate(df, cols, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def read_table(path: str | Path, cols: dict[str, str]) -> pd.DataFrame:
    """Read parquet at `path` and validate/cast it against schema `cols`."""
    return _validate(pd.read_parquet(path), cols, path)


def save_npz(path: str | Path, **arrays) -> Path:
    """np.savez_compressed with parent dirs. scipy.sparse values are stored as CSR parts
    (no pickle) and come back as csr_matrix from load_npz."""
    path = Path(path)
    flat = {}
    for name, a in arrays.items():
        if sp.issparse(a):
            a = sp.csr_matrix(a)
            flat[f"{name}{_SPARSE_TAG}data"] = a.data
            flat[f"{name}{_SPARSE_TAG}indices"] = a.indices
            flat[f"{name}{_SPARSE_TAG}indptr"] = a.indptr
            flat[f"{name}{_SPARSE_TAG}shape"] = np.asarray(a.shape)
        else:
            flat[name] = np.asarray(a)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)
    return path


def load_npz(path: str | Path) -> dict:
    """Inverse of save_npz: returns {name: ndarray | csr_matrix}."""
    out, parts = {}, {}
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            if _SPARSE_TAG in key:
                name, field = key.split(_SPARSE_TAG)
                parts.setdefault(name, {})[field] = z[key]
            else:
                out[key] = z[key]
    for name, p in parts.items():
        out[name] = sp.csr_matrix(
            (p["data"], p["indices"], p["indptr"]), shape=tuple(p["shape"])
        )
    return out


def ensure_dirs() -> None:
    """Create every directory named in contracts.py (relative to the current directory)."""
    for d in (SHOTS_DIR, GRAPHS_DIR, TRACES_DIR, COSTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
