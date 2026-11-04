# OWNER: P3
"""scripts/p3_demo.py: deterministic output, transcript recording, sparkline scaling."""
import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qdec.contracts import QUICK_GRID

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_demo.py"


def load_demo():
    spec = importlib.util.spec_from_file_location("p3_demo", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def small_grid(mod, monkeypatch):
    monkeypatch.setattr(mod, "QUICK_GRID", [dataclasses.replace(w, shots=2000)
                                            for w in QUICK_GRID[:1]])


def test_sparkline_scaling():
    spark = load_demo().sparkline
    assert spark(np.zeros(100), vmax=10) == " " * 60            # empty queue is blank
    line = spark(np.r_[np.zeros(50), np.full(50, 10.0)], vmax=10)
    assert len(line) == 60 and line[0] == " " and line[-1] == "█"
    assert spark(np.array([3.0]), vmax=10) not in ("", " ")     # short input still drawn
    assert spark(np.array([]), vmax=10) == ""
    lo, hi = spark(np.full(10, 2.0), vmax=100), spark(np.full(10, 50.0), vmax=100)
    assert lo[0] < hi[0]                                        # same vmax -> comparable rows


def test_mock_demo_records_transcript(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    mod = load_demo()
    small_grid(mod, monkeypatch)
    out = tmp_path / "rec" / "demo.txt"
    mod.main(["--mock", "--record", str(out)])
    text = out.read_text(encoding="utf-8")
    for needle in ("Act 1", "Act 2", "Act 3", "Takeaway", "full, controller on",
                   "full, controller off", "MOCK DATA", "MODELED"):
        assert needle in text
    assert capsys.readouterr().out.startswith(text.splitlines()[0])


def test_demo_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = load_demo()
    small_grid(mod, monkeypatch)
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    mod.main(["--mock", "--record", str(a)])
    mod.main(["--mock", "--record", str(b)])
    assert a.read_text() == b.read_text()


def test_unknown_tag_exits_with_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = load_demo()
    small_grid(mod, monkeypatch)
    with pytest.raises(SystemExit, match="no data for"):
        mod.main(["--mock", "--tag", "surface_d99_p0.1_s1"])


def test_period_override_is_used(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    mod = load_demo()
    small_grid(mod, monkeypatch)
    mod.main(["--mock", "--period-ns", "12345"])
    assert "one window every 12.3 us (--period-ns)" in capsys.readouterr().out


def test_change_wording():
    change = load_demo().change
    assert change(100.0, 10.0).strip() == "10.0x lower"
    assert change(10.0, 20.0).strip() == "2.0x HIGHER"
