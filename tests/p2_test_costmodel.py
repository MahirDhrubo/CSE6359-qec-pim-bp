# OWNER: P2
"""Day 2: per-iteration cost models, service tables and iter_costs.json on mock traces."""
import json

import numpy as np
import pandas as pd
import pytest

from qdec.contracts import SERVICE_COLS, System
from qdec.costmodel import conventional, pim
from qdec.costmodel.service import (
    build_service_table,
    iter_cost_entries,
    load_costs,
    write_iter_costs,
)
from qdec.mocks import mock_cache, mock_functional

TAG = "surface_d5_p0.001_s2000"
D5 = {"n_vars": 1677, "n_checks": 120, "n_edges": 5004, "max_var_deg": 4, "max_check_deg": 80}
D7 = {"n_vars": 5471, "n_checks": 336, "n_edges": 17156, "max_var_deg": 4, "max_check_deg": 80}
BP_SYSTEMS = {System.BP_CPU_CONV.value, System.PIM_NOCACHE.value}
CACHE_SYSTEMS = {System.PIM_CACHE.value, System.FULL.value}


@pytest.fixture(scope="module")
def costs():
    return load_costs()


@pytest.fixture(scope="module")
def traces():
    return mock_functional(TAG, n=2000), mock_cache(TAG, n=2000)


# ---------------------------------------------------------------- per-iteration models
@pytest.mark.parametrize("model", [pim, conventional])
def test_per_iteration_positive_and_scales(model, costs):
    a, b = model.per_iteration_cost(D5, 6, costs), model.per_iteration_cost(D7, 6, costs)
    assert set(a) == {"ns", "nj", "bytes"}
    assert a["ns"] > 0 and a["nj"] > 0 and a["bytes"] >= 0
    assert b["nj"] > a["nj"]                         # bigger graph, more energy
    wider = model.per_iteration_cost(D5, 8, costs)
    assert wider["nj"] >= a["nj"]


def test_pim_lockstep_latency_and_no_offchip_bytes(costs):
    a, b = pim.per_iteration_cost(D5, 6, costs), pim.per_iteration_cost(D7, 6, costs)
    assert a["ns"] == b["ns"]                        # same max degrees -> same lockstep latency
    assert a["bytes"] == b["bytes"] == 0


def test_pim_beats_conventional_per_iteration(costs):
    for st in (D5, D7):
        assert pim.per_iteration_cost(st, 6, costs)["ns"] < conventional.per_iteration_cost(st, 6, costs)["ns"]


def test_accumulator_wider_than_messages():
    assert pim.accumulator_bits(6, 4) == 9 and pim.accumulator_bits(6, 1) == 7


def test_conventional_switches_to_dram_when_working_set_spills(costs):
    huge = {**D7, "n_edges": 10_000_000, "n_vars": 3_000_000}
    per = conventional.per_iteration_cost(huge, 6, costs)
    moved, ws = conventional.traffic_bytes(huge, 6)
    assert ws > costs["l2_byte"]["capacity_bytes"]
    assert per["ns"] >= moved * costs["dram_read_byte"]["latency_ns"]


# ---------------------------------------------------------------- service table
def test_service_table_schema_and_rows(traces, costs):
    f, c = traces
    s = build_service_table(f, c, D5, costs, cam_entries=1000, key_bits=272)
    pd.testing.assert_frame_equal(s, s.astype(SERVICE_COLS))
    assert set(s["system"]) == BP_SYSTEMS | CACHE_SYSTEMS
    assert (s.groupby("system").size() == len(f)).all()
    assert set(s["path"]) <= {"cache", "bp", "bp+osd"}
    assert (s["latency_ns"] >= 0).all() and (s["energy_nj"] >= 0).all()


def test_bp_latency_matches_contract_repricing_formula(traces, costs):
    """service latency == bp_iters * per_iter_ns + osd_ran * osd_ns, the formula P3 reprices with,
    so an uncapped controller reproduces the service table exactly."""
    f, c = traces
    s = build_service_table(f, c, D5, costs)
    prices = iter_cost_entries(f, c, D5, costs)
    for system in BP_SYSTEMS:
        p = prices[system]
        got = s[s["system"] == system].sort_values("shot_id")["latency_ns"].to_numpy()
        want = f["bp_iters"] * p["per_iter_ns"] + f["osd_ran"] * p["osd_ns"]
        np.testing.assert_allclose(got, want.to_numpy())


def test_cache_systems_skip_bp_on_hits(traces, costs):
    f, c = traces
    s = build_service_table(f, c, D5, costs, cam_entries=1000, key_bits=272)
    hit = c.sort_values("shot_id")["cache_hit"].to_numpy()
    nocache = s[s["system"] == System.PIM_NOCACHE.value].sort_values("shot_id")
    for system in CACHE_SYSTEMS:
        rows = s[s["system"] == system].sort_values("shot_id")
        assert (rows["path"].to_numpy()[hit] == "cache").all()
        assert (rows["path"].to_numpy()[~hit] != "cache").all()
        slow = hit & (f["bp_iters"].to_numpy() > 0)
        assert (rows["latency_ns"].to_numpy()[slow] < nocache["latency_ns"].to_numpy()[slow]).all()
        # a miss pays the cache front end on top of BP
        assert (rows["latency_ns"].to_numpy()[~hit] >= nocache["latency_ns"].to_numpy()[~hit]).all()


def test_bposd_rows_use_measured_latency(traces, costs):
    f, c = traces
    lat = np.random.default_rng(0).lognormal(11, 1, len(f))
    s = build_service_table(f, c, D5, costs, bposd_latency_ns=lat)
    rows = s[s["system"] == System.BPOSD_CPU.value].sort_values("shot_id")
    np.testing.assert_allclose(rows["latency_ns"], lat)
    assert rows["bytes_moved"].isna().all()
    with pytest.raises(ValueError):
        build_service_table(f, c, D5, costs, bposd_latency_ns=lat[:-1])


def test_misaligned_traces_rejected(traces, costs):
    f, c = traces
    with pytest.raises(ValueError):
        build_service_table(f, c.iloc[:-1], D5, costs)
    with pytest.raises(ValueError):
        build_service_table(f, c.assign(tag="other"), D5, costs)


# ---------------------------------------------------------------- iter_costs.json
def test_write_iter_costs_shape(traces, costs, tmp_path):
    f, c = traces
    entries = {TAG: iter_cost_entries(f, c, D5, costs, cam_entries=1000, key_bits=272)}
    path = write_iter_costs(entries, tmp_path / "iter_costs.json")
    got = json.loads(path.read_text())
    assert set(got[TAG]) == BP_SYSTEMS | CACHE_SYSTEMS
    for e in got[TAG].values():
        assert set(e) == {"per_iter_ns", "osd_ns", "cache_ns"} and all(v >= 0 for v in e.values())
    assert got[TAG][System.PIM_CACHE.value]["cache_ns"] > 0
    assert got[TAG][System.PIM_NOCACHE.value]["cache_ns"] == 0


def test_write_iter_costs_rejects_bad_entries(tmp_path):
    with pytest.raises(ValueError):
        write_iter_costs({TAG: {"not_a_system": {"per_iter_ns": 1, "osd_ns": 1, "cache_ns": 0}}},
                         tmp_path / "x.json")
    with pytest.raises(ValueError):
        write_iter_costs({TAG: {"pim_cache": {"per_iter_ns": 1}}}, tmp_path / "x.json")
