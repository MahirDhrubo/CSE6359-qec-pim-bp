# OWNER: P1
"""scripts/p1_quant_sweep.py on a few real TRAIN shots, plus its pick rule."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from qdec.contracts import TRACES_DIR

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p1sweep")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        _load("p1_make_data").main(["--grid", "quick"])
        df = _load("p1_quant_sweep").main(["--shots", "150", "--bits", "6", "--ranges", "1,2"])
        yield tmp, df


def test_sweep_rows_and_csv(ran):
    tmp, df = ran
    assert len(df) == 3 and pd.isna(df.iloc[0]["quant_bits"])  # float reference first
    assert df["tag"].str.endswith("_s1000").all()  # tuned on TRAIN by default
    assert df["n"].eq(150).all() and bool(df.iloc[0]["passes"])
    assert (df["ci_lo"] <= df["ler"]).all() and (df["ler"] <= df["ci_hi"]).all()
    on_disk = pd.read_csv(tmp / TRACES_DIR / f"quant_sweep_{df.iloc[0]['tag']}.csv")
    assert list(on_disk.columns) == list(df.columns) and len(on_disk) == 3


def test_pick_smallest_passing_bits_within_osd_tolerance():
    pick = _load("p1_quant_sweep").pick
    df = pd.DataFrame([
        {"quant_bits": None, "range_x": None, "passes": True, "ler": 0.01, "osd_frac": 0.3},
        {"quant_bits": 3, "range_x": 1.0, "passes": False, "ler": 0.02, "osd_frac": 0.9},
        {"quant_bits": 4, "range_x": 1.0, "passes": True, "ler": 0.012, "osd_frac": 0.6},
        {"quant_bits": 4, "range_x": 2.0, "passes": True, "ler": 0.011, "osd_frac": 0.7},
        {"quant_bits": 6, "range_x": 2.0, "passes": True, "ler": 0.009, "osd_frac": 0.3},
    ])
    # 4-bit configs pass on LER but run OSD 2x as often as float -> rejected by the OSD rule
    best = pick(df)
    assert (best["quant_bits"], best["range_x"]) == (6, 2.0)
    best = pick(df, osd_tol=float("inf"))  # LER-only rule
    assert (best["quant_bits"], best["range_x"]) == (4, 2.0)
    assert pick(df[df["quant_bits"].isna()]) is None
