# OWNER: P2
"""Build the cluster cache on SEED_TRAIN shots and evaluate it on SEED_TEST shots.

Writes TRACES_DIR/cache_<tag>.parquet (CACHE_COLS) and TRACES_DIR/cache_<tag>.json (CAM size and
key width, read by p2_build_costs.py; P2-internal, not a contract file).
With --mock, writes qdec.mocks.mock_cache tables instead.
Codes outside cam.CACHEABLE_CODES (the gross code) get no cache trace: PIM-BP without a cache.
Workloads whose shots/graph are missing are skipped and listed, so a partly generated FULL_GRID
still produces every trace it can.
Each workload runs in its own child process (--jobs at a time): a native crash inside ldpc during
the build (seen once, non-reproducibly, after 18 FULL_GRID workloads in one process) costs one
retry of that workload instead of the whole run, and memory is returned between workloads.
Usage: python scripts/p2_build_cache.py --grid quick|full [--mock] [--max-entries N] [--jobs N]
"""
import argparse
import dataclasses
import json
import subprocess
import sys
import time

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


def write_cache(w, max_entries) -> str:
    """Build + evaluate one workload and write its trace and CAM sidecar (child process body)."""
    cache, df = build_and_evaluate(w, max_entries)
    (TRACES_DIR / f"cache_{w.tag}.json").write_text(json.dumps({
        "cam_entries": cache.size(), "key_bits": cache.key_bits(),
        "max_entries": cache.max_entries, "built_entries": len(cache.entries),
        "dropped_wrong": cache.n_dropped, "reach": cache.boundary.reach,
        "max_cluster_size": cache.max_cluster_size,
        "train_tag": dataclasses.replace(w, seed=SEED_TRAIN).tag,
    }, indent=1))
    return _report(io.write_table(df, CACHE_COLS, TRACES_DIR / f"cache_{w.tag}.parquet"), df)


def _report(path, df, note="") -> str:
    hit = df["cache_hit"]
    return (f"wrote {path}{note}  hit={hit.mean():.1%}  "
            f"fail_on_hits={int(df['fail_cache'].sum())}/{int(hit.sum())}")


def _inputs(w) -> list:
    train_w = dataclasses.replace(w, seed=SEED_TRAIN)
    return [GRAPHS_DIR / f"{w.code}_d{w.distance}_p{w.p:g}.npz",
            SHOTS_DIR / f"{train_w.tag}.npz", SHOTS_DIR / f"{w.tag}.npz"]


def run_children(todo, args) -> list[str]:
    """One child process per workload, args.jobs at a time; a crashed child is retried once."""
    queue = [(w, 0) for w in todo]
    running, failed = [], []
    while queue or running:
        while queue and len(running) < max(args.jobs, 1):
            w, tries = queue.pop(0)
            cmd = [sys.executable, __file__, "--grid", args.grid, "--tag", w.tag,
                   "--max-entries", str(args.max_entries)]
            running.append((w, tries, subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                                       stderr=subprocess.STDOUT, text=True)))
        time.sleep(0.5)
        for item in [r for r in running if r[2].poll() is not None]:
            running.remove(item)
            w, tries, proc = item
            out = proc.stdout.read().strip()
            if proc.returncode == 0:
                print(out, flush=True)
            elif tries == 0:
                print(f"{w.tag}: child exited with {proc.returncode}; retrying once\n{out[-500:]}", flush=True)
                queue.append((w, 1))
            else:
                print(f"{w.tag}: FAILED twice (exit {proc.returncode})\n{out[-500:]}", flush=True)
                failed.append(w.tag)
    return failed


def main(argv=None) -> None:
    from qdec.cache.cam import CACHEABLE_CODES, DEFAULT_CAM_ENTRIES

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--max-entries", type=int, default=DEFAULT_CAM_ENTRIES,
                    help=f"CAM capacity (default {DEFAULT_CAM_ENTRIES}; 0 = every TRAIN key)")
    ap.add_argument("--jobs", type=int, default=4, help="workloads built in parallel")
    ap.add_argument("--tag", help=argparse.SUPPRESS)   # child mode: build this one workload
    args = ap.parse_args(argv)

    io.ensure_dirs()
    grid = QUICK_GRID if args.grid == "quick" else FULL_GRID
    if args.tag:
        print(write_cache(next(w for w in grid if w.tag == args.tag), args.max_entries or None))
        return
    if args.mock:
        from qdec.mocks import mock_cache
        for w in grid:
            df = mock_cache(w.tag, n=w.shots)
            print(_report(io.write_table(df, CACHE_COLS, TRACES_DIR / f"cache_{w.tag}.parquet"),
                          df, " (MOCK)"))
        return

    todo, missing = [], []
    for w in grid:
        if w.code not in CACHEABLE_CODES:
            print(f"{w.tag}: no cache for the {w.code} code (PIM-BP without cache)")
        elif absent := [p.name for p in _inputs(w) if not p.exists()]:
            missing.append(f"{w.tag} ({', '.join(absent)})")
        else:
            todo.append(w)
    failed = run_children(todo, args)
    if missing:
        print(f"p2_build_cache: SKIPPED {len(missing)} workload(s) with missing inputs "
              f"(run `make data`): " + "; ".join(missing))
    if failed:
        print(f"p2_build_cache: FAILED {len(failed)} workload(s): {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"p2_build_cache: SKIPPED, missing input ({e}) (run the upstream make target, or MOCK=1)")
        sys.exit(0)
    except NotImplementedError:
        print("p2_build_cache: not implemented yet")
        sys.exit(0)
