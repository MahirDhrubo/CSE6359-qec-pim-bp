# OWNER: P1
import hashlib
import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from qdec import io
from qdec.contracts import (
    FUNCTIONAL_COLS,
    GRAPHS_DIR,
    QUICK_GRID,
    SEED_TRAIN,
    SHOTS_DIR,
    TRACES_DIR,
)
from qdec.workload.circuits import make_circuit, sample_shots
from qdec.workload.functional import run_functional
from qdec.workload.graphs import dem_to_graph, graph_stats
from qdec.workload.verification import verify_workload


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = replace(QUICK_GRID[0], shots=8)
    graph = dem_to_graph(make_circuit(w))
    gname = f"{w.code}_d{w.distance}_p{w.p:g}"
    gp = io.save_npz(GRAPHS_DIR / f"{gname}.npz", **graph)
    gp.with_suffix(".json").write_text(json.dumps(graph_stats(graph["H"])))
    for ww in (w, replace(w, seed=SEED_TRAIN)):
        io.save_npz(SHOTS_DIR / f"{ww.tag}.npz", **sample_shots(ww))
    df = run_functional(w)
    path = io.write_table(df, FUNCTIONAL_COLS, TRACES_DIR / f"functional_{w.tag}.parquet")
    metadata = {"kind": "FUNCTIONAL", "workload": asdict(w), **df.attrs,
                "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (gp, SHOTS_DIR / f"{w.tag}.npz")}}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return w, path


def test_complete_real_artifacts_pass(dataset):
    w, _ = dataset
    result = verify_workload(w)
    assert result["shots"] == 8 and result["n"] == 8
    assert not result["latency_present"]


def test_partial_count_cannot_be_certified_as_full(dataset):
    w, _ = dataset
    with pytest.raises(ValueError, match="invalid shots"):
        verify_workload(replace(w, shots=20_000))


def test_mock_and_stale_metadata_rejected(dataset):
    w, path = dataset
    p = path.with_suffix(".json")
    metadata = json.loads(p.read_text())
    metadata["kind"] = "MOCK"
    p.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mock"):
        verify_workload(w)


def test_data_only_is_explicit_and_latency_required_on_request(dataset):
    w, _ = dataset
    with pytest.raises(FileNotFoundError):
        verify_workload(w, require_latency=True)
    assert verify_workload(w, data_only=True) == {"tag": w.tag, "shots": 8}


def test_verifier_cli_enters_dataset_and_labels_profile(dataset, monkeypatch, capsys):
    w, _ = dataset
    work = Path.cwd()
    script = Path(__file__).resolve().parents[1] / "scripts/p1_verify_data.py"
    spec = importlib.util.spec_from_file_location("p1_verify_data", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "QUICK_GRID", [w])
    monkeypatch.chdir(work.parent)
    results = mod.main(["--work-dir", str(work)])
    assert len(results) == 1 and results[0]["tag"] == w.tag
    assert "COMPLETE GRID: 1/1" in capsys.readouterr().out
