# OWNER: P1
"""Bounded regeneration check against a frozen 1,000-shot profile.

Uses a temporary directory; never modifies reference artifacts. Checks fresh
TRAIN/TEST inputs and graphs, then 16 evenly spaced TEST shots for each of three
representative workloads. Run after CPU measurements finish. This is not a full
simulation rerun or a new CPU benchmark.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import qdec
from qdec import io
from qdec.contracts import FULL_GRID, FUNCTIONAL_COLS, SEED_TRAIN
from qdec.workload.circuits import make_circuit, sample_shots
from qdec.workload.functional import run_functional
from qdec.workload.graphs import dem_to_graph, graph_stats


def assert_arrays_equal(actual, expected):
    if set(actual) != set(expected):
        raise ValueError('array keys differ')
    for key in actual:
        a, b = actual[key], expected[key]
        if sp.issparse(a) and sp.issparse(b):
            assert a.shape == b.shape and a.dtype == b.dtype, key
            assert (a != b).nnz == 0, key
        else:
            assert a.dtype == b.dtype, key
            np.testing.assert_array_equal(a, b, err_msg=key)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True, help='directory containing data/')
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args(argv)
    reference, report = args.reference.resolve(), args.report.resolve()
    source = Path(qdec.__file__).resolve().parents[2]
    selected = [('surface', 5, .003), ('surface', 13, .005), ('gross', 12, .003)]
    results = []
    original = Path.cwd()
    versions = {n: version(n) for n in ('numpy', 'scipy', 'stim', 'ldpc', 'pymatching', 'pandas')}
    versions['python'] = sys.version.split()[0]
    try:
        with tempfile.TemporaryDirectory(prefix='p1-reproduce-') as scratch:
            os.chdir(scratch)
            for code, distance, p in selected:
                w = replace(next(w for w in FULL_GRID if
                                 (w.code, w.distance, w.p) == (code, distance, p)), shots=1000)
                print(f'checking {w.tag}', flush=True)
                trace = reference / 'data/traces' / f'functional_{w.tag}.parquet'
                meta = json.loads(trace.with_suffix('.json').read_text())
                assert meta['versions'] == versions, 'environment differs from generating environment'
                for name, digest in meta['source_sha256'].items():
                    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
                name = f'{code}_d{distance}_p{p:g}'
                graph = dem_to_graph(make_circuit(w))
                assert_arrays_equal(graph, io.load_npz(reference / 'data/graphs' / f'{name}.npz'))
                assert graph_stats(graph['H']) == json.loads(
                    (reference / 'data/graphs' / f'{name}.json').read_text())
                io.save_npz(Path('data/graphs') / f'{name}.npz', **graph)
                for ww in (replace(w, seed=SEED_TRAIN), w):
                    fresh = sample_shots(ww)
                    assert_arrays_equal(fresh, io.load_npz(reference / 'data/shots' / f'{ww.tag}.npz'))
                ids = np.linspace(0, 999, 16, dtype=int)
                subset = dict(fresh)
                subset['dets'], subset['obs'] = fresh['dets'][ids], fresh['obs'][ids]
                io.save_npz(Path('data/shots') / f'{w.tag}.npz', **subset)
                actual = run_functional(replace(w, shots=len(ids)))
                actual['shot_id'] = ids
                expected = io.read_table(trace, FUNCTIONAL_COLS).iloc[ids].reset_index(drop=True)
                pd.testing.assert_frame_equal(actual, expected)
                results.append({'tag': w.tag, 'input_shots_per_seed': 1000,
                                'decoded_shot_ids': ids.tolist(), 'arrays_and_graph': 'PASS',
                                'all_functional_columns': 'PASS'})
                print(f'PASS {w.tag}: both seeds, graph, and 16 decoded rows', flush=True)
    finally:
        os.chdir(original)
    payload = {'scope': 'partial regeneration; existing environment, separate source checkout',
               'source_root': str(source),
               'source_revision': subprocess.check_output(
                   ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
               'versions': versions, 'results': results}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
