# OWNER: P2
"""Processing-in-SRAM cost of one quantized min-sum BP iteration. MODELED."""


def per_iteration_cost(graph_stats: dict, quant_bits: int, costs: dict) -> dict[str, float]:
    """Cost of one BP iteration with messages resident in SRAM and updated in place
    (sram_xor, sram_min_search, bitserial_add, convergence_or_reduce).

    Inputs:  graph_stats from GRAPHS_DIR/<code>_d<d>_p<p>.json (GRAPH_STATS_KEYS),
             quant_bits from FUNCTIONAL_COLS.quant_bits, costs = parsed op_costs.yaml.
    Outputs: {"ns": latency per iteration, "nj": energy per iteration, "bytes": bytes moved}.
    Contract: feeds SERVICE_COLS rows for PIM_NOCACHE / PIM_CACHE / FULL and iter_costs.json.
    Owner:   P2
    """
    raise NotImplementedError
