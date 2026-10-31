# OWNER: SHARED
"""Contract tests: every mock round-trips through qdec.io against its schema."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from qdec import io
from qdec.contracts import (
    CACHE_COLS,
    FULL_GRID,
    FUNCTIONAL_COLS,
    QUICK_GRID,
    SERVICE_COLS,
    System,
)
from qdec.mocks import mock_cache, mock_functional, mock_service

TAG = QUICK_GRID[0].tag
MOCKS = [
    (mock_functional, FUNCTIONAL_COLS),
    (mock_cache, CACHE_COLS),
    (mock_service, SERVICE_COLS),
]


@pytest.mark.parametrize("mock,cols", MOCKS, ids=lambda x: getattr(x, "__name__", ""))
def test_mock_roundtrip(tmp_path, mock, cols):
    df = mock(TAG, n=500)
    path = io.write_table(df, cols, tmp_path / "sub" / "t.parquet")
    back = io.read_table(path, cols)
    assert list(back.columns) == list(cols)
    assert {c: str(t) for c, t in back.dtypes.items()} == cols
    pd.testing.assert_frame_equal(back, df[list(cols)].astype(cols))


def test_write_table_rejects_wrong_columns(tmp_path):
    df = mock_cache(TAG, n=10)
    with pytest.raises(ValueError, match="missing"):
        io.write_table(df.drop(columns=["cache_hit"]), CACHE_COLS, tmp_path / "t.parquet")
    with pytest.raises(ValueError, match="extra"):
        io.write_table(df.assign(junk=1), CACHE_COLS, tmp_path / "t.parquet")


def test_workload_tags_unique():
    tags = [w.tag for w in FULL_GRID]
    assert len(tags) == len(set(tags))


def test_mock_service_covers_every_system():
    assert set(mock_service(TAG, n=10)["system"]) == {s.value for s in System}


def test_npz_roundtrip_with_sparse(tmp_path):
    H = sp.random(7, 11, density=0.3, format="csr", random_state=0)
    dets = np.random.default_rng(0).random((4, 7)) < 0.5
    io.save_npz(tmp_path / "g.npz", H=H, dets=dets)
    back = io.load_npz(tmp_path / "g.npz")
    assert sp.issparse(back["H"]) and (back["H"] != H).nnz == 0
    np.testing.assert_array_equal(back["dets"], dets)


def test_ensure_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    io.ensure_dirs()
    for d in ("data/shots", "data/graphs", "data/traces", "data/costs", "results"):
        assert (tmp_path / d).is_dir()
