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
from importlib.metadata import version
from pathlib import Path

from qdec import io
from qdec.contracts import FULL_GRID, FUNCTIONAL_COLS, GRAPHS_DIR, QUICK_GRID, SHOTS_DIR, TRACES_DIR
from qdec.workload.selection import select_workloads


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--code", choices=("surface", "gross"))
    ap.add_argument("--max-distance", type=int, help="surface distance cap; retains gross")
    ap.add_argument("--shots", type=int, help="explicit reduced-profile shot cap")
    ap.add_argument("--resume", action="store_true", help="skip only verified matching checkpoints")
    ap.add_argument("--d", type=int, help="optionally select one workload (requires --p)")
    ap.add_argument("--p", type=float, help="optionally select one workload (requires --d)")
    ap.add_argument("--output-dir", type=Path, default=TRACES_DIR,
                    help="optional separate destination for a preserved validation run")
    args = ap.parse_args(argv)

    io.ensure_dirs()
    grid = QUICK_GRID if args.grid == "quick" else FULL_GRID
    try:
        grid = select_workloads(grid, code=args.code, max_distance=args.max_distance, shots=args.shots)
    except ValueError as e:
        ap.error(str(e))
    if (args.d is None) != (args.p is None):
        ap.error("--d and --p must be supplied together")
    if args.d is not None:
        grid = [w for w in grid if w.distance == args.d and w.p == args.p]
        if len(grid) != 1:
            ap.error("selection must match exactly one workload")
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), root / "src/qdec/contracts.py", root / "src/qdec/io.py"]
    sources += [root / "src/qdec/workload" / f"{name}.py" for name in
                ("functional", "baselines", "circuits", "gross", "bp_minsum", "osd", "selection")]
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sources}
    versions = {name: version(name) for name in
                ("numpy", "scipy", "stim", "ldpc", "pymatching", "pandas")}
    versions["python"] = sys.version.split()[0]
    for w in grid:
        print(f"decoding {w.tag}{' (MOCK)' if args.mock else ''}", flush=True)
        inputs = [] if args.mock else [SHOTS_DIR / f"{w.tag}.npz",
                    GRAPHS_DIR / f"{w.code}_d{w.distance}_p{w.p:g}.npz"]
        input_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        path = args.output_dir / f"functional_{w.tag}.parquet"
        if args.resume and (path.exists() or path.with_suffix(".json").exists()):
            if not path.exists() or not path.with_suffix(".json").exists():
                raise ValueError(f"incomplete checkpoint: {path}; use a fresh output directory")
            saved = json.loads(path.with_suffix(".json").read_text())
            if (saved.get("kind") != ("MOCK" if args.mock else "FUNCTIONAL")
                    or saved.get("workload") != dataclasses.asdict(w)
                    or saved.get("input_sha256") != input_hashes
                    or saved.get("source_sha256") != source_hashes
                    or saved.get("versions") != versions
                    or saved.get("output_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
                raise ValueError(f"stale checkpoint: {path}; use a fresh output directory")
            print(f"verified checkpoint, skipping {w.tag}", flush=True)
            continue
        if args.mock:
            from qdec.mocks import mock_functional
            df = mock_functional(w.tag, n=w.shots)
        else:
            from qdec.workload.functional import run_functional
            df = run_functional(w)
        path = io.write_table(df, FUNCTIONAL_COLS, path)
        metadata = {"kind": "MOCK" if args.mock else "FUNCTIONAL",
                    "workload": dataclasses.asdict(w), "input_sha256": input_hashes,
                    "source_sha256": source_hashes,
                    "versions": versions,
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
