# OWNER: P2
"""Build the cluster cache on SEED_TRAIN shots and evaluate it on SEED_TEST shots.

Writes TRACES_DIR/cache_<tag>.parquet (CACHE_COLS) and TRACES_DIR/cache_<tag>.json (CAM size and
key width, read by p2_build_costs.py; P2-internal, not a contract file).
With --mock, writes qdec.mocks.mock_cache tables instead.
Usage: python scripts/p2_build_cache.py --grid quick|full [--mock] [--max-entries N]
"""
import argparse
import dataclasses
import json
import sys

from qdec import io
from qdec.contracts import (
    CACHE_COLS,
    FULL_GRID,
    GRAPHS_DIR,
    QUICK_GRID,
    SEED_TRAIN,
    SHOTS_DIR,
    TRACES_DIR,
)


def build_and_evaluate(w, max_entries=None):
    """(built cache, CACHE_COLS frame) for one SEED_TEST workload."""
    from qdec.cache.cam import ClusterCache

    train_w = dataclasses.replace(w, seed=SEED_TRAIN)
    graph = io.load_npz(GRAPHS_DIR / f"{w.code}_d{w.distance}_p{w.p:g}.npz")
    train = io.load_npz(SHOTS_DIR / f"{train_w.tag}.npz")
    test = io.load_npz(SHOTS_DIR / f"{w.tag}.npz")
    cache = ClusterCache(graph, train["det_coords"], max_entries=max_entries)
    cache.build(train, train_w.tag)
    return cache, cache.evaluate(test, w.tag)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--max-entries", type=int, default=None, help="CAM capacity (default: all keys)")
    args = ap.parse_args(argv)

    io.ensure_dirs()
    grid = QUICK_GRID if args.grid == "quick" else FULL_GRID
    for w in grid:
        if args.mock:
            from qdec.mocks import mock_cache
            df = mock_cache(w.tag, n=w.shots)
        else:
            cache, df = build_and_evaluate(w, args.max_entries)
            (TRACES_DIR / f"cache_{w.tag}.json").write_text(json.dumps({
                "cam_entries": cache.size(), "key_bits": cache.key_bits(),
                "max_entries": cache.max_entries, "train_keys": len(cache.entries),
                "dropped_wrong": cache.n_dropped, "reach": cache.boundary.reach,
                "max_cluster_size": cache.max_cluster_size,
                "train_tag": dataclasses.replace(w, seed=SEED_TRAIN).tag,
            }, indent=1))
        path = io.write_table(df, CACHE_COLS, TRACES_DIR / f"cache_{w.tag}.parquet")
        hit = df["cache_hit"]
        print(f"wrote {path}{' (MOCK)' if args.mock else ''}  hit={hit.mean():.1%}  "
              f"fail_on_hits={int(df['fail_cache'].sum())}/{int(hit.sum())}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"p2_build_cache: SKIPPED, missing input ({e}) (run the upstream make target, or MOCK=1)")
        sys.exit(0)
    except NotImplementedError:
        print("p2_build_cache: not implemented yet")
        sys.exit(0)
