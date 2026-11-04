# OWNER: P2
"""Per-shot service times for every System, plus per-iteration prices for P3's controller."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qdec.contracts import SERVICE_COLS, System
from qdec.costmodel import conventional, pim

REQUIRED_OPS = (
    "sram_xor", "sram_min_search", "bitserial_add", "convergence_or_reduce",
    "cam_search", "dram_read_byte", "dram_write_byte", "l2_byte", "cpu_op", "cpu_core_ns",
    "relay_fpga_iter", "nearmem_dpu_iter", "osd_ns",
)

# Systems priced from bp_iters / osd_ran (cache systems add the cache path on top).
CACHE_SYSTEMS = (System.PIM_CACHE, System.FULL)   # FULL's controller is applied in P3's queue sim
# Published-number baselines: op_costs.yaml entry with the paper's per-iteration cost.
PUBLISHED_ITER_OPS = {System.NEARMEM_DPU: "nearmem_dpu_iter", System.RELAY_FPGA: "relay_fpga_iter"}
# Splitter + key generation on the controller core, scalar instructions per shot:
#   one adjacency test and one isolation-distance test per defect pair, KEY_OPS_PER_DEFECT per
#   defect (grid offset subtract and pack).
KEY_OPS_PER_DEFECT = 4


def load_costs(path: str | Path | None = None) -> dict:
    """Parse op_costs.yaml (default: the file next to this module).

    Outputs: {op_name: {"latency_ns": float, "energy_pj": float, "source": str}}.
             Extra yaml fields (unit, status, note) are passed through.
    Raises ValueError on a missing op, a null / NaN number or a missing / "TODO" source.
             Only a status PUBLISHED entry may leave energy_pj as .nan (paper did not report it).
    Owner:   P2
    """
    path = Path(path) if path else Path(__file__).with_name("op_costs.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [op for op in REQUIRED_OPS if op not in raw]
    if missing:
        raise ValueError(f"{path}: missing ops {missing}")
    out = {}
    for op, e in raw.items():
        nan_ok = {"energy_pj"} if e.get("status") == "PUBLISHED" else set()
        bad = [f for f in ("latency_ns", "energy_pj")
               if type(e.get(f)) not in (int, float) or (math.isnan(e[f]) and f not in nan_ok)]
        if bad:
            raise ValueError(f"{path}: {op}.{bad} must be numbers, got {[e.get(f) for f in bad]}")
        src = str(e.get("source") or "").strip()
        if not src or "TODO" in src:
            raise ValueError(f"{path}: {op}.source needs a citation")
        out[op] = {**e, "latency_ns": float(e["latency_ns"]),
                   "energy_pj": float(e["energy_pj"]), "source": src}
    return out


def io_bytes(graph_stats: dict) -> float:
    """Per-shot bytes into and out of the decoder: syndrome in, correction out (1 bit each)."""
    return float(math.ceil(graph_stats["n_checks"] / 8) + math.ceil(graph_stats["n_vars"] / 8))


def cache_path_cost(n_defects, n_lookups, cam_entries: int, key_bits: int,
                    costs: dict) -> tuple[np.ndarray, np.ndarray]:
    """(ns, nJ) per shot for the cache front end: splitter + keys on the controller core
    (paid by every shot, hit or miss) and n_lookups CAM searches over cam_entries x key_bits."""
    k = np.asarray(n_defects, dtype=np.float64)
    ops = k * np.maximum(k - 1, 0) + KEY_OPS_PER_DEFECT * k
    look = np.asarray(n_lookups, dtype=np.float64)
    cpu, cam = costs["cpu_op"], costs["cam_search"]
    ns = ops * cpu["latency_ns"] + look * cam["latency_ns"]
    pj = ops * cpu["energy_pj"] + look * cam_entries * key_bits * cam["energy_pj"]
    return ns, pj / 1e3


def osd_cost(costs: dict, measured_ns: float | None = None) -> tuple[float, float]:
    """(ns, nJ) of one CPU OSD fallback: the tag's measured time if given, else the yaml
    osd_ns constant. Energy = time on one busy core at cpu_core_ns power."""
    ns = costs["osd_ns"]["latency_ns"] if measured_ns is None else float(measured_ns)
    core = costs["cpu_core_ns"]
    return ns, ns / core["latency_ns"] * core["energy_pj"] / 1e3


def _aligned(functional_df: pd.DataFrame, cache_df: pd.DataFrame | None):
    f = functional_df.sort_values("shot_id", kind="stable").reset_index(drop=True)
    if f["tag"].nunique() != 1 or f["quant_bits"].nunique() != 1:
        raise ValueError("functional_df must hold one tag with one quant_bits")
    if cache_df is None:
        return f, None
    c = cache_df.sort_values("shot_id", kind="stable").reset_index(drop=True)
    if not (f["shot_id"].equals(c["shot_id"]) and (c["tag"] == f["tag"].iloc[0]).all()):
        raise ValueError("functional and cache traces must cover the same tag and shot_ids")
    return f, c


def build_service_table(
    functional_df: pd.DataFrame,
    cache_df: pd.DataFrame | None,
    graph_stats: dict,
    costs: dict,
    *,
    cam_entries: int = 0,
    key_bits: int = 0,
    bposd_latency_ns: np.ndarray | None = None,
    osd_ns: float | None = None,
) -> pd.DataFrame:
    """Combine P1's functional trace and P2's cache trace into per-shot service times.

    Inputs:  functional_df (FUNCTIONAL_COLS, TRACES_DIR/functional_<tag>.parquet),
             cache_df (CACHE_COLS, TRACES_DIR/cache_<tag>.parquet), graph_stats (GRAPH_STATS_KEYS),
             costs from load_costs(); cam_entries / key_bits of the CAM that produced cache_df
             (TRACES_DIR/cache_<tag>.json); bposd_latency_ns = P1's MEASURED ldpc BP+OSD wall
             clock per shot (TRACES_DIR/bposd_latency_<tag>.npz), or None to omit bposd_cpu;
             osd_ns = this tag's measured OSD fallback time (COSTS_DIR/osd_<tag>.json median),
             or None for the op_costs.yaml fallback constant.
    Outputs: DataFrame matching contracts.SERVICE_COLS with one row per shot per System
             (path in {"cache", "bp", "bp+osd"}; bposd_cpu uses MEASURED latency from P1,
             nearmem_dpu / relay_fpga come from published numbers, everything else MODELED).
             BP systems: bp_iters * per_iter + osd_ran * osd (the contracts.py repricing formula
             at full budget). nearmem_dpu / relay_fpga use the paper's per-iteration time on
             the same iteration trace and OSD fallback (status PUBLISHED in op_costs.yaml).
             Cache systems add the splitter + CAM cost to every shot and skip BP on a hit;
             with cache_df=None (code not in cam.CACHEABLE_CODES) FULL = PIM-BP rows and
             PIM_CACHE is omitted. bytes_moved = bytes between storage and a separate compute
             unit (core <-> L2/DRAM for the CPU, only per-shot syndrome I/O for PIM); NaN for
             bposd_cpu and the published baselines, whose traffic is not modeled. path for
             bposd_cpu follows the quantized trace's osd_ran (ldpc's own OSD decisions are not
             in P1's hand-off).
    Contract: written via io.write_table to COSTS_DIR / f"service_{tag}.parquet".
    Owner:   P2
    """
    f, c = _aligned(functional_df, cache_df)
    tag, qb, n = f["tag"].iloc[0], int(f["quant_bits"].iloc[0]), len(f)
    iters = f["bp_iters"].to_numpy(np.float64)
    osd = f["osd_ran"].to_numpy(bool)
    bp_path = np.where(osd, "bp+osd", "bp")
    osd_ns, osd_nj = osd_cost(costs, osd_ns)
    shot_io = io_bytes(graph_stats)

    def bp(per):
        return (iters * per["ns"] + osd * osd_ns, iters * per["nj"] + osd * osd_nj,
                iters * per["bytes"] + shot_io)

    def frame(system, path, lat, nj, nbytes):
        return pd.DataFrame({"shot_id": f["shot_id"], "tag": tag, "system": system.value,
                             "path": path, "latency_ns": lat, "energy_nj": nj,
                             "bytes_moved": nbytes})

    out = []
    if bposd_latency_ns is not None:
        lat = np.asarray(bposd_latency_ns, dtype=np.float64)
        if lat.shape != (n,):
            raise ValueError(f"bposd latency has {lat.shape} entries for {n} shots")
        core = costs["cpu_core_ns"]
        out.append(frame(System.BPOSD_CPU, bp_path, lat,
                         lat / core["latency_ns"] * core["energy_pj"] / 1e3, np.full(n, np.nan)))
    out.append(frame(System.BP_CPU_CONV, bp_path,
                     *bp(conventional.per_iteration_cost(graph_stats, qb, costs))))
    for system, op in PUBLISHED_ITER_OPS.items():
        lat, nj, _ = bp(published_per_iteration(op, costs))
        out.append(frame(system, bp_path, lat, nj, np.full(n, np.nan)))
    pim_lat, pim_nj, pim_bytes = bp(pim.per_iteration_cost(graph_stats, qb, costs))
    out.append(frame(System.PIM_NOCACHE, bp_path, pim_lat, pim_nj, pim_bytes))
    if c is not None:
        hit = c["cache_hit"].to_numpy(bool)
        c_ns, c_nj = cache_path_cost(f["n_defects"], c["n_lookups"], cam_entries, key_bits, costs)
        for system in CACHE_SYSTEMS:
            out.append(frame(system, np.where(hit, "cache", bp_path),
                             c_ns + np.where(hit, 0.0, pim_lat), c_nj + np.where(hit, 0.0, pim_nj),
                             np.where(hit, shot_io, pim_bytes)))
    else:  # no cache for this code: FULL is PIM-BP + controller only, PIM_CACHE does not exist
        out.append(frame(System.FULL, bp_path, pim_lat, pim_nj, pim_bytes))
    return pd.concat(out, ignore_index=True).astype(SERVICE_COLS)


def published_per_iteration(op: str, costs: dict) -> dict[str, float]:
    """Per-iteration cost of a published-number baseline, shaped like pim/conventional's.
    bytes: NaN, the papers do not report traffic."""
    e = costs[op]
    return {"ns": e["latency_ns"], "nj": e["energy_pj"] / 1e3, "bytes": float("nan")}


def iter_cost_entries(
    functional_df: pd.DataFrame,
    cache_df: pd.DataFrame | None,
    graph_stats: dict,
    costs: dict,
    *,
    cam_entries: int = 0,
    key_bits: int = 0,
    osd_ns: float | None = None,
) -> dict[str, dict[str, float]]:
    """{system_value: {"per_iter_ns", "osd_ns", "cache_ns"}} for one tag, for every System with
    a per-iteration price (modeled or published). cache_ns = mean cache-path time of the tag's
    hit shots (0 without a cache). bposd_cpu is MEASURED end to end and has no per-iteration
    split (P1's hand-off rules out deriving OSD time by subtracting medians), so it is omitted."""
    f, c = _aligned(functional_df, cache_df)
    qb = int(f["quant_bits"].iloc[0])
    osd_ns, _ = osd_cost(costs, osd_ns)
    per_iter = {System.BP_CPU_CONV: conventional.per_iteration_cost(graph_stats, qb, costs)["ns"],
                System.PIM_NOCACHE: pim.per_iteration_cost(graph_stats, qb, costs)["ns"],
                **{s: costs[op]["latency_ns"] for s, op in PUBLISHED_ITER_OPS.items()}}
    out = {s.value: {"per_iter_ns": ns, "osd_ns": osd_ns, "cache_ns": 0.0}
           for s, ns in per_iter.items()}
    per_pim = per_iter[System.PIM_NOCACHE]
    if c is not None:
        hit = c["cache_hit"].to_numpy(bool)
        c_ns, _ = cache_path_cost(f["n_defects"][hit], c["n_lookups"][hit], cam_entries,
                                  key_bits, costs)
        cache_ns = float(c_ns.mean()) if hit.any() else 0.0
        for system in CACHE_SYSTEMS:
            out[system.value] = {"per_iter_ns": per_pim, "osd_ns": osd_ns, "cache_ns": cache_ns}
    else:
        out[System.FULL.value] = {"per_iter_ns": per_pim, "osd_ns": osd_ns, "cache_ns": 0.0}
    return out


def write_iter_costs(entries: dict, path: str | Path) -> Path:
    """Write the controller price list.

    Inputs:  {tag: {system_value: {"per_iter_ns": float, "osd_ns": float, "cache_ns": float}}}.
    Outputs: JSON file in exactly that shape (see bottom of contracts.py).
    Contract: COSTS_DIR / "iter_costs.json", read by P3's BacklogController.reprice.
    Owner:   P2
    """
    valid = {s.value for s in System}
    clean = {}
    for tag, systems in entries.items():
        clean[tag] = {}
        for system, e in systems.items():
            if system not in valid:
                raise ValueError(f"{tag}: unknown system {system!r}")
            if set(e) != {"per_iter_ns", "osd_ns", "cache_ns"}:
                raise ValueError(f"{tag}/{system}: keys must be per_iter_ns, osd_ns, cache_ns")
            clean[tag][system] = {k: float(e[k]) for k in ("per_iter_ns", "osd_ns", "cache_ns")}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=1))
    return path
