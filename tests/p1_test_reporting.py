# OWNER: P1
from dataclasses import replace

import numpy as np
import pytest

from qdec.contracts import QUICK_GRID
from qdec.mocks import mock_functional
from qdec.workload.reporting import summarize_functional


def fixture_data():
    w = replace(QUICK_GRID[0], shots=4)
    df = mock_functional(w.tag, n=4)
    df["quant_bits"] = 8
    df["n_defects"] = 1
    df["bp_iters"] = [1, 50, 2, 50]
    df["bp_converged"] = [True, False, True, False]
    df["osd_ran"] = ~df.bp_converged
    df["fail_bp_osd"] = [True, False, False, False]
    df["fail_bp_float"] = [False, True, False, False]
    df["fail_matching"] = False
    return w, df, {"dets": np.ones((4, 1), dtype=bool)}


def test_paired_outcomes_and_tails():
    w, df, shots = fixture_data()
    r = summarize_functional(w, df, shots, 8)
    assert (r["only_q_fails"], r["only_float_fails"]) == (1, 1)
    assert r["quant_osd_frac"] == .5 and r["iters_p99"] == 50
    assert r["quant_ler"] == .25 and r["ler_screen_pass"]


@pytest.mark.parametrize("col,value", [("shot_id", 0), ("tag", "wrong"),
                                      ("quant_bits", 6), ("n_defects", 2),
                                      ("osd_ran", False), ("bp_iters", 51)])
def test_rejects_misaligned_or_invalid_trace(col, value):
    w, df, shots = fixture_data()
    df[col] = value
    with pytest.raises(ValueError):
        summarize_functional(w, df, shots, 8)


def test_rejects_partial_trace():
    w, df, shots = fixture_data()
    with pytest.raises(ValueError, match="complete TEST"):
        summarize_functional(w, df.iloc[:2], shots, 8)
