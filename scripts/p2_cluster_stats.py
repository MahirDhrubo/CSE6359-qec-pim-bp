# OWNER: P2
"""Cluster-count / cluster-size statistics on P2's own Stim shots (Day 1 sanity check).

Also reports two upper bounds for Day 2: the share of TEST shots whose clusters are all
isolated, and the share whose clusters all have a key already seen in TRAIN.
Usage: python scripts/p2_cluster_stats.py [--shots N] [--reach R]
"""
import argparse

import numpy as np

from qdec.cache.devshots import sample_dev_shots
from qdec.cache.keys import Boundary, canonical_key, is_isolated
from qdec.cache.splitter import detector_adjacency, find_clusters
from qdec.contracts import QUICK_GRID, SEED_TEST, SEED_TRAIN


def shot_clusters(s: dict):
    A = detector_adjacency(s["H"])
    return [find_clusters(row, s["det_coords"], A) for row in s["dets"]]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=int, default=20_000)
    ap.add_argument("--reach", type=int, default=2)
    args = ap.parse_args(argv)

    print(f"{'workload':<16} {'mean_cl':>7} {'p99_cl':>6} {'max_sz':>6} {'sz=1':>6} {'sz<=2':>6} "
          f"{'empty':>6} {'isolated':>8} {'seen':>6} {'keys':>6}")
    for w in sorted({(w.distance, w.p) for w in QUICK_GRID}):
        d, p = w
        train = sample_dev_shots(d, p, args.shots, SEED_TRAIN)
        test = sample_dev_shots(d, p, args.shots, SEED_TEST)
        bd = Boundary.from_coords(train["det_coords"], reach=args.reach,
                                  clamp=max(2, args.reach))
        train_keys = {canonical_key(c, bd) for cl in shot_clusters(train) for c in cl}

        cls = shot_clusters(test)
        counts = np.array([len(cl) for cl in cls])
        sizes = np.array([len(c.det_ids) for cl in cls for c in cl])
        iso = np.mean([all(is_isolated(c, cl, bd) for c in cl) for cl in cls])
        seen = np.mean([all(canonical_key(c, bd) in train_keys for c in cl) for cl in cls])
        print(f"d={d} p={p:<9g} {counts.mean():7.2f} {np.percentile(counts, 99):6.0f} "
              f"{sizes.max():6d} {np.mean(sizes == 1):6.0%} {np.mean(sizes <= 2):6.0%} "
              f"{np.mean(counts == 0):6.0%} {iso:8.0%} {seen:6.0%} {len(train_keys):6d}")
    print("isolated = TEST shots whose clusters all pass is_isolated; "
          "seen = all cluster keys appear in TRAIN (hit-rate upper bounds before Day 2 build)")


if __name__ == "__main__":
    main()
