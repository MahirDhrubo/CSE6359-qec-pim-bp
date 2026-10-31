# OWNER: P2
"""Build the cluster cache on SEED_TRAIN shots and evaluate it on SEED_TEST shots.

Writes TRACES_DIR/cache_<tag>.parquet (CACHE_COLS).
With --mock, writes qdec.mocks.mock_cache tables instead.
Usage: python scripts/p2_build_cache.py --grid quick|full [--mock]
"""
import argparse
import dataclasses
import sys

from qdec import io
from qdec.contracts import CACHE_COLS, FULL_GRID, QUICK_GRID, SEED_TRAIN, SHOTS_DIR, TRACES_DIR


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args(argv)

    io.ensure_dirs()
    grid = QUICK_GRID if args.grid == "quick" else FULL_GRID
    for w in grid:
        if args.mock:
            from qdec.mocks import mock_cache
            df = mock_cache(w.tag, n=w.shots)
        else:
            from qdec.cache.cam import ClusterCache
            cache = ClusterCache()
            train = io.load_npz(SHOTS_DIR / f"{dataclasses.replace(w, seed=SEED_TRAIN).tag}.npz")
            test = io.load_npz(SHOTS_DIR / f"{w.tag}.npz")
            cache.build(train)
            df = cache.evaluate(test)
        path = io.write_table(df, CACHE_COLS, TRACES_DIR / f"cache_{w.tag}.parquet")
        print(f"wrote {path}{' (MOCK)' if args.mock else ''}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"p2_build_cache: SKIPPED, missing input ({e}) (run the upstream make target, or MOCK=1)")
        sys.exit(0)
    except NotImplementedError:
        print("p2_build_cache: not implemented yet")
        sys.exit(0)
