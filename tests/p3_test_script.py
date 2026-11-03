# OWNER: P3
"""End-to-end smoke test of scripts/p3_make_results.py --mock (run in a temp directory)."""
import dataclasses
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from qdec.contracts import QUICK_GRID, System

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_make_results.py"


def load_script():
    spec = importlib.util.spec_from_file_location("p3_make_results", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mock_run_writes_every_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = load_script()
    monkeypatch.setattr(mod, "QUICK_GRID", [dataclasses.replace(w, shots=3000)
                                            for w in QUICK_GRID[:2]])
    mod.main(["--mock"])
    out = tmp_path / "results" / "mock"
    rates = pd.read_csv(out / "max_rate.csv")
    assert set(rates["system"]) == {s.value for s in System}
    assert (rates["rate_hz"] <= rates["rate_stable_hz"] + 1e-6).all()   # tail criterion is stricter
    assert len(pd.read_csv(out / "summary.csv")) == 2 * len(System)
    for name in ("max_rate.png", "backlog_explosion.png",
                 f"latency_cdf_{QUICK_GRID[0].tag}.png"):
        assert (out / name).stat().st_size > 10_000
    assert (tmp_path / "results" / "backlog_explosion_synthetic.png").exists()


def test_real_mode_without_inputs_raises_file_not_found(tmp_path, monkeypatch):
    """main() raises; the __main__ guard turns this into the SKIPPED message and exit 0."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_script().main([])
