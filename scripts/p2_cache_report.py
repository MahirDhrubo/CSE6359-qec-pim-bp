# OWNER: P2
"""Day 2 cache report: hit rate vs p and d, CAM size vs hit rate, and LER on hits.

Builds each workload's cache once on SEED_TRAIN, then re-evaluates SEED_TEST at every CAM
capacity in --sizes (most frequent TRAIN keys kept). When P1's functional trace exists, compares
fail_cache with quantized BP+OSD (fail_bp_osd) on the same hit shots (95% Wilson CIs).
Writes data/p2/cache_sweep.csv and data/p2/cache_sweep.png (P2 scratch, not a contract file).
Usage: python scripts/p2_cache_report.py [--grid quick|full] [--sizes 16 64 256 ...]
"""
import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from qdec import io
from qdec.cache.cam import ClusterCache
from qdec.contracts import (
    DATA,
    FULL_GRID,
    FUNCTIONAL_COLS,
    GRAPHS_DIR,
    QUICK_GRID,
    SEED_TRAIN,
    SHOTS_DIR,
    TRACES_DIR,
)

OUT = DATA / "p2"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    ph, den = k / n, 1 + z * z / n
    mid, half = (ph + z * z / (2 * n)) / den, z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half) if k else 0.0, min(1.0, mid + half))


def sweep(w, sizes) -> list[dict]:
    train_w = dataclasses.replace(w, seed=SEED_TRAIN)
    graph = io.load_npz(GRAPHS_DIR / f"{w.code}_d{w.distance}_p{w.p:g}.npz")
    train = io.load_npz(SHOTS_DIR / f"{train_w.tag}.npz")
    test = io.load_npz(SHOTS_DIR / f"{w.tag}.npz")
    cache = ClusterCache(graph, train["det_coords"])
    cache.build(train, train_w.tag)
    splits = cache.split(test)
    fpath = TRACES_DIR / f"functional_{w.tag}.parquet"
    fdf = io.read_table(fpath, FUNCTIONAL_COLS).sort_values("shot_id") if fpath.exists() else None

    rows = []
    for size in [*sizes, None]:
        if size is not None and size >= len(cache.entries):
            continue
        cache.resize(size)
        df = cache.evaluate(test, w.tag, splits)
        hit = df["cache_hit"].to_numpy(bool)
        nonempty = df["n_clusters"].to_numpy() > 0
        k_c = int(df["fail_cache"][hit].sum())
        row = {"tag": w.tag, "d": w.distance, "p": w.p, "cam_entries": cache.size(),
               "key_bits": cache.key_bits(), "hit_rate": hit.mean(),
               "hit_rate_nonempty": hit[nonempty].mean(), "hits": int(hit.sum()),
               "fail_cache": k_c, "ler_cache_hits": k_c / max(hit.sum(), 1)}
        if fdf is not None:
            k_b = int(fdf["fail_bp_osd"].to_numpy(bool)[hit].sum())
            row.update(fail_bposd_on_hits=k_b, ler_bposd_hits=k_b / max(hit.sum(), 1),
                       ler_cache_ci_hi=wilson(k_c, int(hit.sum()))[1],
                       ler_bposd_ci_hi=wilson(k_b, int(hit.sum()))[1],
                       ler_bposd_all=fdf["fail_bp_osd"].mean())
        rows.append(row)
    return rows


def plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for tag, g in df.groupby("tag", sort=False):
        ax.plot(g["cam_entries"], g["hit_rate_nonempty"], marker="o", label=tag)
    ax.set(xscale="log", xlabel="CAM entries (most frequent TRAIN keys)",
           ylabel="hit rate (shots with >= 1 defect)", ylim=(0, 1),
           title="DecodeCache: CAM size vs hit rate (TEST shots)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("quick", "full"), default="quick")
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 256, 1024, 4096, 16384])
    args = ap.parse_args(argv)

    grid = QUICK_GRID if args.grid == "quick" else FULL_GRID
    rows = []
    for w in grid:
        rows += sweep(w, args.sizes)
        print(f"done {w.tag}", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "cache_sweep.csv", index=False)
    plot(df, OUT / "cache_sweep.png")

    full = df.groupby("tag", sort=False).tail(1)
    print("\nHit rate vs p and d (full CAM):")
    cols = ["d", "p", "cam_entries", "key_bits", "hit_rate", "hit_rate_nonempty", "hits", "fail_cache"]
    if "fail_bposd_on_hits" in df:
        cols += ["fail_bposd_on_hits", "ler_cache_ci_hi", "ler_bposd_ci_hi"]
    print(full[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\nCAM size vs hit rate (shots with >= 1 defect):")
    print(df.pivot(index="cam_entries", columns="tag", values="hit_rate_nonempty")
          .to_string(float_format=lambda x: f"{x:.3f}", na_rep="-"))
    if "fail_bposd_on_hits" in df:
        ok = (full["fail_cache"] <= full["fail_bposd_on_hits"]).all()
        print(f"\nfail_cache LER <= BP+OSD LER on hits (full CAM, every tag): {'PASS' if ok else 'FAIL'}")
    print(f"wrote {OUT / 'cache_sweep.csv'} and {OUT / 'cache_sweep.png'}")


if __name__ == "__main__":
    main()
