# OWNER: P1
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "p1_summarize_sweep", Path(__file__).resolve().parents[1] / "scripts/p1_summarize_sweep.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_histogram_conserves_shots_and_distinguishes_maximum():
    histogram = mod.iteration_histogram([0, 1, 5, 6, 10, 11, 25, 26, 49, 50, 50])
    assert histogram == {"0": 1, "1–5": 2, "6–10": 2, "11–25": 2, "26–49": 2, "50": 2}
    with pytest.raises(ValueError):
        mod.iteration_histogram([51])


def test_missing_sweep_requires_explicit_partial_label(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="Incomplete"):
        mod.main(["--grid", "full", "--shots", "1000"])
    assert mod.main(["--grid", "full", "--shots", "1000", "--allow-partial"]) == []
    out = capsys.readouterr().out
    assert "REDUCED PROFILE; PARTIAL: 0/23" in out
