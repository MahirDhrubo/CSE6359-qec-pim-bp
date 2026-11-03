# OWNER: P2
"""Conventional (CPU + memory hierarchy data movement) cost of one quantized min-sum BP
iteration. MODELED.

Software layout: messages stored one per byte (int8, quant_bits <= 8), edge lists as int32.
Per iteration the check pass and the variable pass each read and write every edge message and
read the edge index, the variable pass also reads the channel LLRs and writes hard decisions.
Bytes are priced at L2 cost while the working set fits l2_byte.capacity_bytes, else at DRAM
cost. Compute and memory overlap (roofline): latency = max(ops, bytes) time; energy adds up.
"""
import math

INDEX_BYTES = 4      # int32 edge -> node index
# scalar instructions per edge per iteration:
#   check: |m| and sign, min1 and min2 compares, sign product, select min1|min2, apply sign,
#          normalize (shift-add); var: accumulate, extrinsic subtract, saturate; syndrome: xor
OPS_PER_EDGE = 12


def traffic_bytes(graph_stats: dict, quant_bits: int) -> tuple[float, float]:
    """(bytes moved per iteration, working-set bytes)."""
    msg = math.ceil(quant_bits / 8)
    E, V = graph_stats["n_edges"], graph_stats["n_vars"]
    moved = 4 * E * msg + 2 * E * INDEX_BYTES + V * msg + V
    working_set = 2 * E * msg + E * INDEX_BYTES + 2 * V * msg
    return float(moved), float(working_set)


def per_iteration_cost(graph_stats: dict, quant_bits: int, costs: dict) -> dict[str, float]:
    """Cost of one BP iteration when messages stream between DRAM/cache and the CPU.

    Inputs:  graph_stats from GRAPHS_DIR/<code>_d<d>_p<p>.json (GRAPH_STATS_KEYS),
             quant_bits from FUNCTIONAL_COLS.quant_bits, costs = parsed op_costs.yaml.
    Outputs: {"ns": latency per iteration, "nj": energy per iteration, "bytes": bytes moved}.
    Contract: feeds SERVICE_COLS rows for System.BP_CPU_CONV and iter_costs.json.
    Owner:   P2
    """
    moved, working_set = traffic_bytes(graph_stats, quant_bits)
    l2 = costs["l2_byte"]
    if working_set <= l2.get("capacity_bytes", 0):
        mem_ns, mem_pj = l2["latency_ns"], l2["energy_pj"]
    else:  # reads and writes stream at the same rate; average their energies
        r, w = costs["dram_read_byte"], costs["dram_write_byte"]
        mem_ns, mem_pj = max(r["latency_ns"], w["latency_ns"]), (r["energy_pj"] + w["energy_pj"]) / 2
    ops = OPS_PER_EDGE * graph_stats["n_edges"]
    cpu = costs["cpu_op"]
    ns = max(ops * cpu["latency_ns"], moved * mem_ns)
    pj = ops * cpu["energy_pj"] + moved * mem_pj
    return {"ns": float(ns), "nj": pj / 1e3, "bytes": moved}
