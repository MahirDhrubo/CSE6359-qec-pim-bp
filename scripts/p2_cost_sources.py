# OWNER: P2
"""Write src/qdec/costmodel/SOURCES.md (cost-model slide table + full citations) from op_costs.yaml.

Re-run after every op_costs.yaml change; tests/p2_test_op_costs.py fails while it is stale.
Usage: python scripts/p2_cost_sources.py
"""
from qdec.costmodel.sources import SOURCES_MD, render


def main() -> None:
    SOURCES_MD.write_text(render(), encoding="utf-8")
    print(f"wrote {SOURCES_MD}")


if __name__ == "__main__":
    main()
