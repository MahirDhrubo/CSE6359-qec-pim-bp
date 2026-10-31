# OWNER: P1
"""scripts/p1_check_baselines.py on a few real shots: writes the agreed T9 latency hand-off."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from qdec import io
from qdec.contracts import QUICK_GRID, TRACES_DIR

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p1check")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        _load("p1_make_data").main(["--grid", "quick"])
        yield tmp, _load("p1_check_baselines").main(["--shots", "300"])


def test_default_is_d5_p1e3_with_both_decoders(ran):
    _, results = ran
    assert [r["tag"] for r in results] == [QUICK_GRID[0].tag]
    r = results[0]
    assert r["n"] == 300 and {"bposd", "matching", "cis_overlap"} <= set(r)
    for name in ("bposd", "matching"):
        lo, hi = r[name]["ci"]
        assert lo <= r[name]["ler"] <= hi


def test_latency_handoff_files(ran):
    tmp, _ = ran
    tag = QUICK_GRID[0].tag
    z = io.load_npz(tmp / TRACES_DIR / f"bposd_latency_{tag}.npz")
    assert set(z) == {"shot_id", "latency_ns"}
    assert z["shot_id"].dtype == np.int64 and z["latency_ns"].dtype == np.float64
    np.testing.assert_array_equal(z["shot_id"], np.arange(300))
    assert (z["latency_ns"] > 0).all()

    info = json.loads((tmp / TRACES_DIR / f"bposd_latency_{tag}.json").read_text())
    assert info["label"] == "MEASURED" and info["threads"] == 1 and info["cpu"]
    assert info["tag"] == tag and info["n_shots"] == 300 and info["max_iters"] == 50
