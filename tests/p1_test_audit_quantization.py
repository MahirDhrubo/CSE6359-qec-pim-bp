# OWNER: P1
import importlib.util
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from qdec.contracts import QUICK_GRID, SEED_TRAIN

spec = importlib.util.spec_from_file_location(
    "p1_audit_quantization", Path(__file__).resolve().parents[1]
    / "scripts/p1_audit_quantization.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def save(directory, w, *, bits=8, osd=.31, n=None, tag=None):
    w = replace(w, seed=SEED_TRAIN)
    common = {"tag": tag or w.tag, "n": n or w.shots, "fails": 5, "p_worse": 1.0}
    rows = [dict(common, quant_bits=None, range_x=None, osd_frac=.3),
            dict(common, quant_bits=bits, range_x=1.0, osd_frac=osd)]
    path = directory / f"quant_sweep_{w.tag}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_missing_is_not_passing_and_all_grid_required(tmp_path):
    save(tmp_path, QUICK_GRID[0])
    summary = audit_module.summarize(audit_module.audit(tmp_path, bits=(7, 8), ranges=(1,)))
    assert summary.verdict.tolist() == ["INCOMPLETE", "INCOMPLETE"]
    assert summary.missing.tolist() == [4, 3]
    for w in QUICK_GRID[1:]:
        save(tmp_path, w)
    summary = audit_module.summarize(audit_module.audit(tmp_path, bits=(7, 8), ranges=(1,)))
    assert summary.verdict.tolist() == ["INCOMPLETE", "PASS ALL"]


def test_one_osd_failure_rejects_even_with_missing_workloads(tmp_path):
    save(tmp_path, QUICK_GRID[0], osd=.4)
    summary = audit_module.summarize(audit_module.audit(tmp_path, bits=(8,), ranges=(1,)))
    assert summary.verdict.tolist() == ["REJECTED"]


@pytest.mark.parametrize("kwargs", [{"n": 300}, {"tag": QUICK_GRID[0].tag}])
def test_reject_partial_or_test_evidence(tmp_path, kwargs):
    save(tmp_path, QUICK_GRID[0], **kwargs)
    with pytest.raises(ValueError, match="TRAIN shots"):
        audit_module.audit(tmp_path)


def test_recomputes_ci_and_does_not_trust_saved_pass_flag(tmp_path):
    path = save(tmp_path, QUICK_GRID[0])
    df = pd.read_csv(path)
    df.loc[1, "fails"] = 1000
    df["passes"] = True
    df.to_csv(path, index=False)
    table = audit_module.audit(tmp_path, bits=(8,), ranges=(1,))
    assert table.iloc[0].status == "FAIL"


def test_reject_duplicate_configuration(tmp_path):
    path = save(tmp_path, QUICK_GRID[0])
    df = pd.read_csv(path)
    pd.concat([df, df.iloc[[1]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique configs"):
        audit_module.audit(tmp_path)
