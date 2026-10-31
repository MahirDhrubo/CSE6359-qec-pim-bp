"""
qdec/contracts.py — the ONLY shared interface between the three branches.

Frozen on `main` at hour 0. Any change = tiny PR to main that all three approve,
then everyone runs `git rebase main`. Never edit this file inside a feature branch.

Data flow (every arrow is a file on disk, so nobody imports anyone else's code):

  P1 workload ──shots/*.npz, graphs/*.npz, graphs/*.json──► P2 cache + cost model
  P1 workload ──traces/functional_<tag>.parquet───────────► P2 cost model
  P2 cache    ──traces/cache_<tag>.parquet────────────────► P2 cost model
  P2 costs    ──costs/service_<tag>.parquet + iter_costs.json──► P3 queue sim + plots
  P1 + P2 traces ─────────────────────────────────────────► P3 LER / hit-rate plots
"""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------- paths
DATA = Path("data")                 # git-ignored; regenerate with `make data`
SHOTS_DIR = DATA / "shots"          # P1 writes   <tag>.npz : dets[bool S×D], obs[bool S×O], det_coords[float D×k]
GRAPHS_DIR = DATA / "graphs"        # P1 writes   <code>_d<d>_p<p>.npz : H (csr), L (csr), priors ; .json : GRAPH_STATS_KEYS
TRACES_DIR = DATA / "traces"        # P1: functional_<tag>.parquet   P2: cache_<tag>.parquet
COSTS_DIR = DATA / "costs"          # P2: service_<tag>.parquet, iter_costs.json
RESULTS_DIR = Path("results")       # P3: figures + tables (committed at the end)

# ---------------------------------------------------------------- seeds
SEED_TRAIN = 1000                   # cache is built ONLY from these shots
SEED_TEST = 2000                    # every system is evaluated on these exact shots

# ---------------------------------------------------------------- workloads
@dataclass(frozen=True)
class Workload:
    code: str          # "surface" | "gross"
    distance: int      # gross code: 12
    p: float           # physical error rate
    rounds: int        # syndrome rounds; one shot = one decode window (4-day simplification)
    shots: int
    seed: int

    @property
    def tag(self) -> str:
        return f"{self.code}_d{self.distance}_p{self.p:g}_s{self.seed}"


# Small grid first (Day 1-2); FULL_GRID on Day 3 only.
QUICK_GRID = [Workload("surface", d, p, d, 20_000, SEED_TEST) for d in (5, 7) for p in (1e-3, 3e-3)]
FULL_GRID = (
    [Workload("surface", d, p, d, 100_000, SEED_TEST) for d in (5, 7, 9, 11, 13) for p in (1e-3, 2e-3, 3e-3, 5e-3)]
    + [Workload("gross", 12, p, 12, 50_000, SEED_TEST) for p in (1e-3, 2e-3, 3e-3)]
)

GRAPH_STATS_KEYS = ("n_vars", "n_checks", "n_edges", "max_var_deg", "max_check_deg")

# ---------------------------------------------------------------- systems (one per baseline row)
class System(str, Enum):
    BPOSD_CPU = "bposd_cpu"              # ldpc BP+OSD, measured wall clock
    BP_CPU_CONV = "bp_cpu_conventional"  # same quantized BP, data-movement model
    NEARMEM_DPU = "nearmem_dpu"          # published numbers only
    RELAY_FPGA = "relay_fpga"            # published numbers only
    PIM_NOCACHE = "pim_nocache"
    PIM_CACHE = "pim_cache"              # no controller
    FULL = "full"                        # pim + cache + backlog controller (applied in P3's queue sim)

# ---------------------------------------------------------------- table schemas
# P1 -> traces/functional_<tag>.parquet  (one row per test shot)
FUNCTIONAL_COLS = {
    "shot_id": "int64",
    "tag": "string",
    "n_defects": "int32",
    "quant_bits": "int8",
    "bp_iters": "int32",        # iterations run; == max_iters when not converged
    "bp_converged": "bool",
    "osd_ran": "bool",
    "fail_bp_osd": "bool",      # logical failure, quantized BP+OSD
    "fail_bp_float": "bool",    # logical failure, unquantized BP+OSD (reference)
    "fail_matching": "boolean", # PyMatching, surface code only (nullable)
}

# P2 -> traces/cache_<tag>.parquet  (one row per test shot)
CACHE_COLS = {
    "shot_id": "int64",
    "tag": "string",
    "n_clusters": "int32",
    "max_cluster_size": "int32",
    "n_lookups": "int32",
    "cache_hit": "bool",        # ALL clusters hit AND passed isolation check
    "fail_cache": "boolean",    # logical failure on the cache path; null when miss
}

# P2 -> costs/service_<tag>.parquet  (one row per shot per System)
SERVICE_COLS = {
    "shot_id": "int64",
    "tag": "string",
    "system": "string",         # System value
    "path": "string",           # "cache" | "bp" | "bp+osd"
    "latency_ns": "float64",    # MODELED unless system == bposd_cpu
    "energy_nj": "float64",
    "bytes_moved": "float64",
}

# P2 -> costs/iter_costs.json : {tag: {system: {"per_iter_ns": f, "osd_ns": f, "cache_ns": f}}}
# P3's backlog controller uses these to re-price a shot when it caps the iteration budget:
#   capped_latency = min(bp_iters, budget) * per_iter_ns + (osd_ns if bp_iters > budget or osd_ran else 0)
