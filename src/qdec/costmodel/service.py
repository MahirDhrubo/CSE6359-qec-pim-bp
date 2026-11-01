# OWNER: P2
"""Per-shot service times for every System, plus per-iteration prices for P3's controller."""
from pathlib import Path

import pandas as pd
import yaml

from qdec.contracts import SERVICE_COLS, System  # noqa: F401

REQUIRED_OPS = (
    "sram_xor", "sram_min_search", "bitserial_add", "convergence_or_reduce",
    "cam_search", "dram_read_byte", "dram_write_byte", "cpu_op", "osd_ns",
)


def load_costs(path: str | Path | None = None) -> dict:
    """Parse op_costs.yaml (default: the file next to this module).

    Outputs: {op_name: {"latency_ns": float, "energy_pj": float, "source": str}}.
             Extra yaml fields (unit, status, note) are passed through.
    Raises ValueError on a missing op, a null number or a missing / "TODO" source.
    Owner:   P2
    """
    path = Path(path) if path else Path(__file__).with_name("op_costs.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [op for op in REQUIRED_OPS if op not in raw]
    if missing:
        raise ValueError(f"{path}: missing ops {missing}")
    out = {}
    for op, e in raw.items():
        bad = [f for f in ("latency_ns", "energy_pj")
               if type(e.get(f)) not in (int, float)]
        if bad:
            raise ValueError(f"{path}: {op}.{bad} must be numbers, got {[e.get(f) for f in bad]}")
        src = str(e.get("source") or "").strip()
        if not src or "TODO" in src:
            raise ValueError(f"{path}: {op}.source needs a citation")
        out[op] = {**e, "latency_ns": float(e["latency_ns"]),
                   "energy_pj": float(e["energy_pj"]), "source": src}
    return out


def build_service_table(
    functional_df: pd.DataFrame,
    cache_df: pd.DataFrame,
    graph_stats: dict,
    costs: dict,
) -> pd.DataFrame:
    """Combine P1's functional trace and P2's cache trace into per-shot service times.

    Inputs:  functional_df (FUNCTIONAL_COLS, TRACES_DIR/functional_<tag>.parquet),
             cache_df (CACHE_COLS, TRACES_DIR/cache_<tag>.parquet), graph_stats (GRAPH_STATS_KEYS),
             costs from load_costs().
    Outputs: DataFrame matching contracts.SERVICE_COLS with one row per shot per System
             (path in {"cache", "bp", "bp+osd"}; bposd_cpu uses MEASURED latency from P1,
             nearmem_dpu / relay_fpga come from published numbers, everything else MODELED).
    Contract: written via io.write_table to COSTS_DIR / f"service_{tag}.parquet".
    Owner:   P2
    """
    raise NotImplementedError


def write_iter_costs(entries: dict, path: str | Path) -> Path:
    """Write the controller price list.

    Inputs:  {tag: {system_value: {"per_iter_ns": float, "osd_ns": float, "cache_ns": float}}}.
    Outputs: JSON file in exactly that shape (see bottom of contracts.py).
    Contract: COSTS_DIR / "iter_costs.json", read by P3's BacklogController.reprice.
    Owner:   P2
    """
    raise NotImplementedError
