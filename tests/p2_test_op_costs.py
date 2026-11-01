# OWNER: P2
"""Day 1: op_costs.yaml is complete, cited and loadable."""
import pytest

from qdec.costmodel.service import REQUIRED_OPS, load_costs


def test_every_op_has_numbers_and_source():
    costs = load_costs()
    assert set(REQUIRED_OPS) <= set(costs)
    for op, e in costs.items():
        assert e["latency_ns"] > 0 and e["energy_pj"] > 0, op
        assert e["source"] and "TODO" not in e["source"], op


def test_no_null_or_todo_left_in_yaml():
    from pathlib import Path

    import qdec.costmodel as cm
    text = (Path(cm.__file__).parent / "op_costs.yaml").read_text(encoding="utf-8")
    assert "null" not in text and '"TODO"' not in text


@pytest.mark.parametrize("bad", [
    "sram_xor: {latency_ns: null, energy_pj: 1, source: x}",
    "sram_xor: {latency_ns: 1, energy_pj: 1, source: \"TODO\"}",
])
def test_load_costs_rejects_unfilled(tmp_path, bad):
    path = tmp_path / "c.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError):
        load_costs(path)
