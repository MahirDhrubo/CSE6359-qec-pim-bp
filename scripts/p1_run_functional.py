# OWNER: P1
"""Run the functional simulation for every workload; writes TRACES_DIR/functional_<tag>.parquet.

With --mock, writes qdec.mocks.mock_functional tables instead (lets P2 test file loading).
Usage: python scripts/p1_run_functional.py --grid quick|full [--mock]
"""
import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

from qdec import io
from qdec.contracts import FULL_GRID, FUNCTIONAL_COLS, GRAPHS_DIR, QUICK_GRID, SHOTS_DIR, TRACES_DIR


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=TRACES_DIR,
                    help="optional separate destination for a preserved validation run")
    args = ap.parse_args(argv)

    io.ensure_dirs()
    grid = QUICK_GRID if args.grid == "quick" else FULL_GRID
    root = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__).resolve(),
                               *sorted((root / "src/qdec/workload").glob("*.py"))]}
    for w in grid:
        print(f"decoding {w.tag}{' (MOCK)' if args.mock else ''}", flush=True)
        inputs = [] if args.mock else [SHOTS_DIR / f"{w.tag}.npz",
                    GRAPHS_DIR / f"{w.code}_d{w.distance}_p{w.p:g}.npz"]
        input_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        if args.mock:
            from qdec.mocks import mock_functional
            df = mock_functional(w.tag, n=w.shots)
        else:
            from qdec.workload.functional import run_functional
            df = run_functional(w)
        path = io.write_table(df, FUNCTIONAL_COLS,
                              args.output_dir / f"functional_{w.tag}.parquet")
        metadata = {"kind": "MOCK" if args.mock else "FUNCTIONAL",
                    "workload": dataclasses.asdict(w), "input_sha256": input_hashes,
                    "source_sha256": source_hashes,
                    "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    **df.attrs}
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        print(f"wrote {path}{' (MOCK)' if args.mock else ''}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"p1_run_functional: missing input ({e}) (run the upstream make target, or MOCK=1)")
        sys.exit(1)
    except NotImplementedError:
        print("p1_run_functional: not implemented yet")
        sys.exit(1)
