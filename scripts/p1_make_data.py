# OWNER: P1
"""Sample shots (SEED_TRAIN and SEED_TEST) and build decoding graphs for every workload.

Writes SHOTS_DIR/<tag>.npz, GRAPHS_DIR/<code>_d<d>_p<p>.npz and .json.
Usage: python scripts/p1_make_data.py --grid quick|full [--mock]
"""
import argparse
import dataclasses
import json
import sys

from qdec import io
from qdec.contracts import FULL_GRID, GRAPHS_DIR, QUICK_GRID, SEED_TRAIN, SHOTS_DIR


def graph_name(w) -> str:
    return f"{w.code}_d{w.distance}_p{w.p:g}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--mock", action="store_true", help="no mock exists for raw shots")
    ap.add_argument("--code", choices=("surface", "gross"), help="optional code-family subset")
    args = ap.parse_args(argv)
    if args.mock:
        print("p1_make_data: there is no mock for shots/graphs; nothing to do.")
        return

    from qdec.workload.circuits import make_circuit, sample_shots
    from qdec.workload.graphs import dem_to_graph, graph_stats

    io.ensure_dirs()
    grid = QUICK_GRID if args.grid == "quick" else FULL_GRID
    if args.code:
        grid = [w for w in grid if w.code == args.code]
        if not grid:
            ap.error("selected code is absent from this grid")
    for w in grid:
        for ww in (w, dataclasses.replace(w, seed=SEED_TRAIN)):
            io.save_npz(SHOTS_DIR / f"{ww.tag}.npz", **sample_shots(ww))
        g = dem_to_graph(make_circuit(w))
        io.save_npz(GRAPHS_DIR / f"{graph_name(w)}.npz", **g)
        (GRAPHS_DIR / f"{graph_name(w)}.json").write_text(json.dumps(graph_stats(g["H"])))
        print(f"wrote {w.tag}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print(f"p1_make_data: incomplete generation: {e}")
        sys.exit(1)
