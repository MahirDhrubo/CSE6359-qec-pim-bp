# OWNER: P1
"""Read-only verification of P1's file-based workload artifacts."""
import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import scipy.sparse as sp

from qdec import io
from qdec.contracts import FUNCTIONAL_COLS, GRAPHS_DIR, SEED_TRAIN, SHOTS_DIR, TRACES_DIR
from qdec.workload.graphs import graph_stats
from qdec.workload.reporting import summarize_functional


def verify_workload(w, *, data_only=False, require_latency=False):
    gname = f"{w.code}_d{w.distance}_p{w.p:g}"
    graph_path = GRAPHS_DIR / f"{gname}.npz"
    g = io.load_npz(graph_path)
    if set(g) != {"H", "L", "priors"}:
        raise ValueError("graph keys do not match the contract")
    h, l, priors = g["H"], g["L"], g["priors"]
    if (not sp.isspmatrix_csr(h) or not sp.isspmatrix_csr(l)
            or l.shape[0] != (1 if w.code == "surface" else 12)
            or l.shape[1] != h.shape[1] or priors.shape != (h.shape[1],)
            or not np.isfinite(priors).all() or not ((priors > 0) & (priors < 1)).all()
            or not np.isin(h.data, [0, 1]).all() or not np.isin(l.data, [0, 1]).all()):
        raise ValueError("invalid decoding graph")
    if json.loads((GRAPHS_DIR / f"{gname}.json").read_text()) != graph_stats(h):
        raise ValueError("graph statistics do not match H")
    for ww in (replace(w, seed=SEED_TRAIN), w):
        shots = io.load_npz(SHOTS_DIR / f"{ww.tag}.npz")
        if (set(shots) != {"dets", "obs", "det_coords"}
                or shots["dets"].shape != (w.shots, h.shape[0])
                or shots["obs"].shape != (w.shots, l.shape[0])
                or shots["dets"].dtype != bool or shots["obs"].dtype != bool
                or shots["det_coords"].shape[0] != h.shape[0]
                or not np.isfinite(shots["det_coords"]).all()):
            raise ValueError(f"invalid shots for {ww.tag}")
    result = {"tag": w.tag, "shots": w.shots}
    if data_only:
        return result
    path = TRACES_DIR / f"functional_{w.tag}.parquet"
    metadata = json.loads(path.with_suffix(".json").read_text())
    if (metadata.get("kind") != "FUNCTIONAL" or metadata.get("workload") != asdict(w)
            or metadata.get("output_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError("functional metadata is missing, mock, or stale")
    for p in (graph_path, SHOTS_DIR / f"{w.tag}.npz"):
        if metadata.get("input_sha256", {}).get(str(p)) != hashlib.sha256(p.read_bytes()).hexdigest():
            raise ValueError(f"functional input hash mismatch: {p}")
    df = io.read_table(path, FUNCTIONAL_COLS)
    result.update(summarize_functional(w, df, shots, metadata["decoder_settings"]["quant_bits"]))
    if w.code == "gross" and df.fail_matching.notna().any():
        raise ValueError("gross matching must be null")
    lat = TRACES_DIR / f"bposd_latency_{w.tag}.npz"
    result["latency_present"] = lat.exists()
    if require_latency or lat.exists():
        z = io.load_npz(lat)
        if (not np.array_equal(z["shot_id"], np.arange(w.shots))
                or z["latency_ns"].shape != (w.shots,)
                or not np.isfinite(z["latency_ns"]).all() or not (z["latency_ns"] > 0).all()):
            raise ValueError("CPU latency is incomplete or invalid")
        info = json.loads(lat.with_suffix(".json").read_text())
        if info.get("label") != "MEASURED" or info.get("n_shots") != w.shots:
            raise ValueError("CPU latency provenance is invalid")
    return result

