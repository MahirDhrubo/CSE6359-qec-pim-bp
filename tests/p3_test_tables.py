# OWNER: P3
import numpy as np
import pandas as pd
import pytest

from qdec.contracts import QUICK_GRID, System
from qdec.mocks import mock_service
from qdec.results.tables import summary_table


def test_quantiles_match_numpy_on_known_data():
    x = np.arange(1, 10_001, dtype=float)          # 1..10000 ns
    df = pd.DataFrame({"tag": "t", "system": "s", "latency_ns": x})
    row = summary_table(df).iloc[0]
    assert row["n"] == 10_000
    assert row["mean_ns"] == pytest.approx(5000.5)
    assert row["p50_ns"] == pytest.approx(np.quantile(x, 0.5))
    assert row["p99_ns"] == pytest.approx(np.quantile(x, 0.99))
    assert row["p999_ns"] == pytest.approx(np.quantile(x, 0.999))
    assert row["max_ns"] == 10_000


def test_one_row_per_tag_and_system_in_system_order():
    tags = [w.tag for w in QUICK_GRID[:2]]
    df = pd.concat([mock_service(t, n=500) for t in tags], ignore_index=True)
    out = summary_table(df)
    assert len(out) == 2 * len(System)
    assert list(out["system"][: len(System)]) == [s.value for s in System]
    assert (out["p50_ns"] <= out["p99_ns"]).all() and (out["p99_ns"] <= out["p999_ns"]).all()


def test_heavier_tail_has_larger_p999_at_same_mean():
    from qdec.queue.synthetic import same_mean_lognormal
    df = same_mean_lognormal(900.0, {"light": 0.3, "heavy": 1.5}, 20_000, seed=0).assign(tag="t")
    out = summary_table(df).set_index("system")
    assert out.loc["light", "mean_ns"] == pytest.approx(out.loc["heavy", "mean_ns"])
    assert out.loc["heavy", "p999_ns"] > 5 * out.loc["light", "p999_ns"]
