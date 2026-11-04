# OWNER: P2
"""Render op_costs.yaml as SOURCES.md: the cost-model slide table plus every full citation.

scripts/p2_cost_sources.py writes the file; tests/p2_test_op_costs.py fails if it is stale.
"""
import math
from pathlib import Path

import yaml

YAML = Path(__file__).with_name("op_costs.yaml")
SOURCES_MD = Path(__file__).with_name("SOURCES.md")

# Model-level assumptions that are not per-op numbers (keep in sync with the module docstrings).
ASSUMPTIONS = [
    ("**PIM-BP** (`pim.py`): bit-serial, transposed Neural Cache layout on 256x256 SRAM subarrays; "
    "all subarrays step in lockstep, so latency follows the max check/variable degree. Accumulators "
    "are wider than the 6-bit messages. Moving messages between check and variable layouts assumes "
    "a network that routes a full 256-bit row per step (optimistic)."),
    ("**CPU baseline** (`conventional.py`): int8 messages, int32 edge lists, 12 scalar instructions "
    "per edge per iteration; bytes are priced at L2 cost while the working set fits 1 MB, else at "
    "DRAM cost; latency = max(compute, memory) (roofline)."),
    ("**Cache front end** (`service.cache_path_cost`): splitter + keys on the controller core "
    "(k(k-1) + 4k scalar instructions for k defects, paid by every shot) plus one CAM search per "
    "looked-up cluster; search energy = entries x key bits x energy per bit."),
    ("**Published baselines** (`nearmem_dpu`, `relay_fpga`): the paper's time per BP iteration "
    "applied to our iteration counts, with the same CPU OSD fallback as every other system."),
    "**Gross code**: no cache (translation symmetry untested); FULL = PIM-BP + controller.",
]
MEASURED = [
    ("**bposd_cpu**: ldpc BP+OSD wall clock per shot, one thread (P1, `p1_check_baselines.py`, "
    "TRACES_DIR/bposd_latency_<tag>.npz/.json)."),
    ("**OSD fallback** (every BP system): median of paired per-shot ldpc timings, "
    "t(BP+OSD) - t(BP) on the same non-converged syndrome (P2, `costmodel/osd_timing.py`, "
    "COSTS_DIR/osd_<tag>.json). The `osd_ns` row below is only the fallback constant."),
    "CPU energy for both = time x 10 W (`cpu_core_ns`, assumed).",
]


def _num(x: float, unit: str) -> str:
    return "not reported" if math.isnan(x) else f"{x:g} {unit}"


def render(path: Path = YAML) -> str:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    lines = [
        "# Cost-model sources",
        "",
        "Generated from `op_costs.yaml` by `scripts/p2_cost_sources.py`; do not edit by hand",
        "(`tests/p2_test_op_costs.py` fails when this file is stale).",
        "",
        "Labels: **MODELED** = our model built on the cited per-operation number; **PUBLISHED** = the",
        "paper's own reported number; **MEASURED** = timed on our CPU. Status VERIFY = value proposed",
        "from the source and still to be checked against the cited table; CHECKED = read off the source.",
        "",
        "## Slide table",
        "",
        "| Operation | Latency | Energy | Unit | Source | Status |",
        "|---|---|---|---|---|---|",
    ]
    for op, e in raw.items():
        lines.append(f"| `{op}` | {_num(float(e['latency_ns']), 'ns')} | "
                     f"{_num(float(e['energy_pj']), 'pJ')} | {e.get('unit', '')} | "
                     f"{e['cite']} | {e.get('status', '')} |")
    lines += ["", "## Measured on our CPU", ""] + [f"- {m}" for m in MEASURED]
    lines += ["", "## Modeling assumptions", ""] + [f"- {a}" for a in ASSUMPTIONS]
    lines += ["", "## Full citations", ""]
    for op, e in raw.items():
        note = f" *Note:* {e['note']}" if e.get("note") else ""
        lines.append(f"- **`{op}`**: {e['source']}.{note}")
    return "\n".join(lines) + "\n"
