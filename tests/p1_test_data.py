# OWNER: P1
"""`make data` end to end: run scripts/p1_make_data.py in a temp dir and read every file back."""
import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from qdec import io
from qdec.contracts import GRAPH_STATS_KEYS, GRAPHS_DIR, QUICK_GRID, SEED_TRAIN, SHOTS_DIR
from qdec.workload.circuits import make_circuit, sample_shots
from qdec.workload.graphs import dem_to_graph

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p1_make_data.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("p1_make_data", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    # contracts paths are relative to the cwd, so run the script inside a temp dir
    tmp = tmp_path_factory.mktemp("p1data")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        mod = _load_script()
        mod.main(["--grid", "quick"])
        yield tmp, mod


@pytest.mark.parametrize("w", QUICK_GRID, ids=lambda w: w.tag)
def test_shots_written_for_both_seeds_and_round_trip(made, w):
    tmp, _ = made
    for ww in (w, dataclasses.replace(w, seed=SEED_TRAIN)):
        z = io.load_npz(tmp / SHOTS_DIR / f"{ww.tag}.npz")
        assert set(z) == {"dets", "obs", "det_coords"}
        want = sample_shots(ww)
        for k in z:
            assert z[k].dtype == want[k].dtype
            np.testing.assert_array_equal(z[k], want[k])
        assert z["dets"].shape[0] == w.shots


@pytest.mark.parametrize("w", QUICK_GRID, ids=lambda w: w.tag)
def test_graphs_round_trip(made, w):
    tmp, mod = made
    z = io.load_npz(tmp / GRAPHS_DIR / f"{mod.graph_name(w)}.npz")
    want = dem_to_graph(make_circuit(w))
    assert set(z) == {"H", "L", "priors"}
    for k in ("H", "L"):
        assert sp.issparse(z[k]) and z[k].dtype == np.uint8
        assert (z[k] != want[k]).nnz == 0
    np.testing.assert_array_equal(z["priors"], want["priors"])

    stats = json.loads((tmp / GRAPHS_DIR / f"{mod.graph_name(w)}.json").read_text())
    assert tuple(stats) == GRAPH_STATS_KEYS
    assert (stats["n_checks"], stats["n_vars"]) == z["H"].shape


def test_graph_name_matches_tag_format():
    w = QUICK_GRID[0]
    assert _load_script().graph_name(w) == f"{w.code}_d{w.distance}_p{w.p:g}"
    assert w.tag.startswith(_load_script().graph_name(w) + "_s")


def test_train_and_test_files_differ(made):
    tmp, _ = made
    w = QUICK_GRID[0]
    test = io.load_npz(tmp / SHOTS_DIR / f"{w.tag}.npz")["dets"]
    train = io.load_npz(tmp / SHOTS_DIR / f"{dataclasses.replace(w, seed=SEED_TRAIN).tag}.npz")
    assert not np.array_equal(test, train["dets"])


def test_code_filter_and_missing_family(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = _load_script()
    monkeypatch.setattr(mod, "FULL_GRID", [dataclasses.replace(QUICK_GRID[0], shots=10)])
    mod.main(["--grid", "full", "--code", "surface"])
    assert (SHOTS_DIR / f"{QUICK_GRID[0].tag}.npz").exists()
    with pytest.raises(SystemExit):
        mod.main(["--grid", "quick", "--code", "gross"])
