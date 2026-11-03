# OWNER: P2
"""Processing-in-SRAM cost of one quantized min-sum BP iteration. MODELED.

Layout (Neural Cache style, bit-serial, transposed): each bitline of a 256 x 256 subarray holds
one node and its messages sit in successive rows, one bit per row. Check nodes and variable nodes
use separate subarrays. Every subarray runs the same row step in lockstep, so latency follows the
largest degree (short nodes are padded) and energy follows the number of active subarrays.

Per iteration, with b = quant_bits, m = b - 1 magnitude bits, a = accumulator bits,
dc / dv = max check / variable degree:
  check:   sign product (dc - 1 xor) + min1/min2 (2 (dc - 1) m min-search steps)
           + per-edge select min1|min2 and apply sign (dc (m + 1) xor)
           + normalize by alpha = 0.625 = x/2 + x/8 (one m-bit add per edge: dc (m + 1) add steps)
           + syndrome check: XOR of the dc hard decisions and the syndrome bit (dc xor)
  var:     posterior = llr + dv messages (dv (a + 1) add) + extrinsic subtract (dv (a + 1) add)
           + saturate back to b bits (dv a xor: OR of the high bits, then select)
  shuffle: every message moves var->check and check->var, plus one hard-decision bit per edge;
           each 256-bit row costs a read + a write row step. ASSUMES a permutation network that
           routes a full row per step (optimistic).
  converge: one convergence_or_reduce per check subarray.
Messages stay resident, so off-chip bytes per iteration are 0 (per-shot syndrome I/O is added
in service.py).
"""
import math

ROW_BITS = 256  # bitlines per subarray; the unit of the sram_* ops in op_costs.yaml


def accumulator_bits(quant_bits: int, max_var_deg: int) -> int:
    """Width that holds llr + max_var_deg messages of quant_bits each without overflow (P1's
    hand-off: posteriors are not stored at message width)."""
    return quant_bits + math.ceil(math.log2(max_var_deg + 1))


def step_counts(graph_stats: dict, quant_bits: int) -> dict[str, dict[str, int]]:
    """Row steps per iteration by phase and op (lockstep, so these are also latency counts)."""
    b, dc, dv = quant_bits, graph_stats["max_check_deg"], graph_stats["max_var_deg"]
    m, a = b - 1, accumulator_bits(quant_bits, graph_stats["max_var_deg"])
    return {
        "check": {"sram_xor": (dc - 1) + dc * (m + 1) + dc,
                  "sram_min_search": 2 * (dc - 1) * m,
                  "bitserial_add": dc * (m + 1)},
        "var": {"bitserial_add": 2 * dv * (a + 1), "sram_xor": dv * a},
        "shuffle": {"sram_xor": 2 * ((dc + dv) * b + dc)},
        "converge": {"convergence_or_reduce": 1},
    }


def per_iteration_cost(graph_stats: dict, quant_bits: int, costs: dict) -> dict[str, float]:
    """Cost of one BP iteration with messages resident in SRAM and updated in place
    (sram_xor, sram_min_search, bitserial_add, convergence_or_reduce).

    Inputs:  graph_stats from GRAPHS_DIR/<code>_d<d>_p<p>.json (GRAPH_STATS_KEYS),
             quant_bits from FUNCTIONAL_COLS.quant_bits, costs = parsed op_costs.yaml.
    Outputs: {"ns": latency per iteration, "nj": energy per iteration, "bytes": bytes moved}.
    Contract: feeds SERVICE_COLS rows for PIM_NOCACHE / PIM_CACHE / FULL and iter_costs.json.
    Owner:   P2
    """
    steps = step_counts(graph_stats, quant_bits)
    n_sub = {"check": math.ceil(graph_stats["n_checks"] / ROW_BITS),
             "var": math.ceil(graph_stats["n_vars"] / ROW_BITS),
             "converge": math.ceil(graph_stats["n_checks"] / ROW_BITS)}
    ns = sum(n * costs[op]["latency_ns"] for ph in steps.values() for op, n in ph.items())
    pj = sum(n * n_sub[ph] * costs[op]["energy_pj"]
             for ph, ops in steps.items() if ph != "shuffle" for op, n in ops.items())
    # shuffle energy: only the rows that actually move (2b + 1 bits per edge), read + write
    rows = math.ceil(graph_stats["n_edges"] * (2 * quant_bits + 1) / ROW_BITS)
    pj += 2 * rows * costs["sram_xor"]["energy_pj"]
    return {"ns": float(ns), "nj": pj / 1e3, "bytes": 0.0}
