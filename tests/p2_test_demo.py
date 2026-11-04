# OWNER: P2
"""Day 4: the P2 live demo runs from scratch (no data/) and writes a valid service table."""
import importlib.util
from pathlib import Path

from qdec import io
from qdec.contracts import SERVICE_COLS, System

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p2_demo.py"


def test_demo_runs_without_data(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)                         # fresh clone: no data/ directory
    spec = importlib.util.spec_from_file_location("p2_demo", SCRIPT)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    demo.main(["--train-shots", "3000", "--test-shots", "400", "--osd-budget-s", "1"])
    out = capsys.readouterr().out
    for section in ("CACHE HIT", "MISS -> BP", "=== all 400 TEST shots ===", "demo finished"):
        assert section in out
    svc = io.read_table(tmp_path / "data/p2/demo/service_surface_d5_p0.001_s2000.parquet",
                        SERVICE_COLS)
    assert set(svc["system"]) == {s.value for s in System}
    assert (svc.groupby("system").size() == 400).all()
