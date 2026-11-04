# OWNER: P2
"""Day 1: op_costs.yaml is complete, cited and loadable."""
import math

import pytest

from qdec.costmodel.service import REQUIRED_OPS, load_costs


def test_every_op_has_numbers_and_source():
    costs = load_costs()
    assert set(REQUIRED_OPS) <= set(costs)
    for op, e in costs.items():
        assert e["latency_ns"] > 0, op
        # only a published baseline may leave energy unreported (.nan)
        assert e["energy_pj"] > 0 or (e["status"] == "PUBLISHED" and math.isnan(e["energy_pj"])), op
        assert e["source"] and "TODO" not in e["source"], op


def test_every_op_has_a_short_cite():
    for op, e in load_costs().items():
        assert str(e.get("cite") or "").strip(), op


def test_sources_md_is_up_to_date():
    """SOURCES.md feeds the cost-model slide; regenerate with scripts/p2_cost_sources.py."""
    from qdec.costmodel.sources import SOURCES_MD, render
    assert SOURCES_MD.read_text(encoding="utf-8") == render(), "run scripts/p2_cost_sources.py"


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


@pytest.mark.parametrize("status, ok", [("VERIFY", False), ("CHECKED", False), ("PUBLISHED", True)])
def test_nan_energy_only_for_published(tmp_path, status, ok):
    from pathlib import Path

    import yaml

    import qdec.costmodel as cm
    raw = yaml.safe_load((Path(cm.__file__).parent / "op_costs.yaml").read_text(encoding="utf-8"))
    raw["sram_xor"].update(energy_pj=float("nan"), status=status)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    if ok:
        assert math.isnan(load_costs(path)["sram_xor"]["energy_pj"])
    else:
        with pytest.raises(ValueError, match="sram_xor"):
            load_costs(path)
