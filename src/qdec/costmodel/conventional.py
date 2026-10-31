# OWNER: P2
"""Conventional (CPU + DRAM data movement) cost of one quantized min-sum BP iteration. MODELED."""


def per_iteration_cost(graph_stats: dict, quant_bits: int, costs: dict) -> dict[str, float]:
    """Cost of one BP iteration when messages stream between DRAM/cache and the CPU.

    Inputs:  graph_stats from GRAPHS_DIR/<code>_d<d>_p<p>.json (GRAPH_STATS_KEYS),
             quant_bits from FUNCTIONAL_COLS.quant_bits, costs = parsed op_costs.yaml.
    Outputs: {"ns": latency per iteration, "nj": energy per iteration, "bytes": bytes moved}.
    Contract: feeds SERVICE_COLS rows for System.BP_CPU_CONV and iter_costs.json.
    Owner:   P2
    """
    raise NotImplementedError
