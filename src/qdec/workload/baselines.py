# OWNER: P1
"""Reference decoders: ldpc (v2) BP+OSD and PyMatching, with measured CPU wall-clock latency."""
import platform
import subprocess
import time
from importlib.metadata import version

import numpy as np
import scipy.sparse as sp
from ldpc import BpOsdDecoder

# Reference BP+OSD settings (T9 hand-off with P2). max_iters is shared with the PIM cost model.
MAX_ITERS = 50
OSD_ORDER = 7
BP_METHOD = "minimum_sum"
OSD_METHOD = "osd_cs"
ALPHA = 0.625  # min-sum scaling factor; re-tuned on Day 2


def bposd_decode_batch(
    H: sp.csr_matrix,
    L: sp.csr_matrix,
    priors: np.ndarray,
    dets: np.ndarray,
    obs: np.ndarray,
    max_iters: int,
    osd_order: int,
    *,
    alpha: float = ALPHA,
    bp_method: str = BP_METHOD,
    osd_method: str = OSD_METHOD,
) -> dict[str, np.ndarray]:
    """Decode every shot with ldpc.BpOsdDecoder and time each decode.

    Inputs:  graph (H, L, priors) from GRAPHS_DIR; dets bool[S x D], obs bool[S x O] from SHOTS_DIR.
    Outputs: {"fail": bool[S], "bp_iters": int[S], "bp_converged": bool[S], "osd_ran": bool[S],
              "latency_ns": float[S]  (MEASURED wall clock, time.perf_counter_ns per shot)}.
    Contract: fail -> FUNCTIONAL_COLS.fail_bp_float; latency_ns -> P2's bposd_cpu service rows.
    Owner:   P1

    Timing covers only decoder.decode(syndrome), one shot at a time on one thread, after one
    warm-up decode. ldpc runs OSD exactly when BP did not converge, so osd_ran = ~bp_converged.
    """
    dec = BpOsdDecoder(
        sp.csr_matrix(H, dtype=np.uint8),
        error_channel=list(np.asarray(priors, dtype=np.float64)),
        max_iter=max_iters,
        bp_method=bp_method,
        ms_scaling_factor=alpha,
        osd_method=osd_method,
        osd_order=osd_order,
        omp_thread_count=1,
    )
    syndromes = np.ascontiguousarray(dets, dtype=np.uint8)
    S = syndromes.shape[0]
    corrections = np.zeros((S, H.shape[1]), dtype=np.uint8)
    bp_iters = np.zeros(S, dtype=np.int32)
    converged = np.zeros(S, dtype=bool)
    latency_ns = np.zeros(S, dtype=np.float64)

    if S:
        dec.decode(syndromes[0])  # warm-up, not timed
    clock = time.perf_counter_ns
    for i in range(S):
        t0 = clock()
        corrections[i] = dec.decode(syndromes[i])
        latency_ns[i] = clock() - t0
        if not syndromes[i].any():
            # ldpc returns early on an empty syndrome and leaves iter/converge from the last shot
            converged[i], bp_iters[i] = True, 0
        else:
            converged[i] = dec.converge
            bp_iters[i] = dec.iter if dec.converge else max_iters

    return {
        "fail": _logical_fail(L, corrections, obs),
        "bp_iters": bp_iters,
        "bp_converged": converged,
        "osd_ran": ~converged,
        "latency_ns": latency_ns,
    }


def _logical_fail(L: sp.csr_matrix, corrections: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """True where the correction's predicted observable flips differ from the real ones."""
    predicted = (sp.csr_matrix(L, dtype=np.uint8) @ corrections.T % 2).T.astype(bool)
    return (predicted != np.asarray(obs, dtype=bool)).any(axis=1)


def measurement_info(**decoder_settings) -> dict:
    """Provenance for MEASURED latencies (written next to bposd_latency_<tag>.npz)."""
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # not macOS: try Linux, then platform
        try:
            with open("/proc/cpuinfo") as f:
                cpu = next(ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name"))
        except (OSError, StopIteration):
            cpu = platform.processor() or platform.machine()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "label": "MEASURED",
        "cpu": cpu,
        "threads": 1,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ldpc": version("ldpc"),
        "stim": version("stim"),
        "pymatching": version("pymatching"),
        "git_commit": commit,
        "timer": "time.perf_counter_ns around decoder.decode, one shot, after one warm-up",
        **decoder_settings,
    }


def matching_decode_batch(circuit, dets: np.ndarray, obs: np.ndarray) -> dict[str, np.ndarray]:
    """Decode every shot with PyMatching (surface code only) and time each decode.

    Inputs:  stim.Circuit (for its detector error model); dets, obs as above.
    Outputs: {"fail": bool[S], "latency_ns": float[S] (MEASURED wall clock)}.
    Contract: fail -> FUNCTIONAL_COLS.fail_matching (null for the gross code).
    Owner:   P1
    """
    raise NotImplementedError
