# OWNER: P1
"""Run the functional simulation for every workload; writes TRACES_DIR/functional_<tag>.parquet.

With --mock, writes qdec.mocks.mock_functional tables instead (lets P2 test file loading).
Usage: python scripts/p1_run_functional.py --grid quick|full [--mock]
"""
import argparse
import sys

from qdec import io
from qdec.contracts import FULL_GRID, FUNCTIONAL_COLS, QUICK_GRID, TRACES_DIR


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args(argv)

    io.ensure_dirs()
    grid = QUICK_GRID if args.grid == "quick" else FULL_GRID
    for w in grid:
        if args.mock:
            from qdec.mocks import mock_functional
            df = mock_functional(w.tag, n=w.shots)
        else:
            from qdec.workload.functional import run_functional
            df = run_functional(w)
        path = io.write_table(df, FUNCTIONAL_COLS, TRACES_DIR / f"functional_{w.tag}.parquet")
        print(f"wrote {path}{' (MOCK)' if args.mock else ''}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"p1_run_functional: SKIPPED, missing input ({e}) (run the upstream make target, or MOCK=1)")
        sys.exit(0)
    except NotImplementedError:
        print("p1_run_functional: not implemented yet")
        sys.exit(0)
