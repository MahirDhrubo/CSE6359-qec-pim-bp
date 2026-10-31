# OWNER: P1
import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qdec import io
from qdec.contracts import FUNCTIONAL_COLS, GRAPHS_DIR, QUICK_GRID, SHOTS_DIR, TRACES_DIR
from qdec.workload.baselines import MAX_ITERS
from qdec.workload.circuits import make_circuit, sample_shots
from qdec.workload.functional import QUANT_BITS, run_functional
from qdec.workload.graphs import dem_to_graph

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    # tag does not include the shot count, so 300 shots stand in for the real file
    w = dataclasses.replace(QUICK_GRID[1], shots=300)  # d=5, p=3e-3
    tmp = tmp_path_factory.mktemp("p1func")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        s = sample_shots(w)
        io.save_npz(SHOTS_DIR / f"{w.tag}.npz", **s)
        io.save_npz(GRAPHS_DIR / f"{w.code}_d{w.distance}_p{w.p:g}.npz",
                    **dem_to_graph(make_circuit(w)))
        yield tmp, w, s, run_functional(w)


def test_schema_and_one_row_per_shot(small):
    _, w, s, df = small
    assert list(df.columns) == list(FUNCTIONAL_COLS)
    assert {c: str(t) for c, t in df.dtypes.items()} == FUNCTIONAL_COLS
    assert len(df) == w.shots
    np.testing.assert_array_equal(df["shot_id"], np.arange(w.shots))
    assert (df["tag"] == w.tag).all() and (df["quant_bits"] == QUANT_BITS).all()
    np.testing.assert_array_equal(df["n_defects"], s["dets"].sum(axis=1))


def test_bp_columns_follow_the_contract(small):
    *_, df = small
    assert (df["bp_iters"][~df["bp_converged"]] == MAX_ITERS).all()
    assert (df["osd_ran"] == ~df["bp_converged"]).all()
    empty = df["n_defects"] == 0
    assert (df["bp_iters"][empty] == 0).all() and df["bp_converged"][empty].all()
    assert df["osd_ran"].any()  # p=3e-3 exercises the OSD path


def test_failures_plausible(small):
    _, _, s, df = small
    raw = s["obs"].any(axis=1).mean()
    for c in ("fail_bp_osd", "fail_bp_float", "fail_matching"):
        assert df[c].mean() < 0.05 < raw  # decoding beats doing nothing
    assert df["fail_matching"].notna().all()  # surface code: matching always reported


def test_written_through_io_roundtrip(small):
    tmp, w, _, df = small
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        path = io.write_table(df, FUNCTIONAL_COLS, TRACES_DIR / f"functional_{w.tag}.parquet")
        back = io.read_table(path, FUNCTIONAL_COLS)
    pd.testing.assert_frame_equal(back, df)


def test_script_mock_mode_still_works(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "p1_run_functional", ROOT / "scripts" / "p1_run_functional.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--grid", "quick", "--mock"])
    for w in QUICK_GRID:
        io.read_table(tmp_path / TRACES_DIR / f"functional_{w.tag}.parquet", FUNCTIONAL_COLS)
