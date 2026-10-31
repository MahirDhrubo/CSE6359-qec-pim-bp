# OWNER: P2
"""Content-addressable cluster cache (CAM model): canonical key -> cached local correction."""
import pandas as pd

from qdec.contracts import CACHE_COLS  # noqa: F401


class ClusterCache:
    """Cache of cluster corrections built from SEED_TRAIN shots only.

    Owner: P2
    """

    def __init__(self, max_entries: int | None = None):
        raise NotImplementedError

    def build(self, train_shots: dict) -> None:
        """Populate the cache from training shots.

        Inputs:  dict from io.load_npz(SHOTS_DIR / f"{tag}.npz") for a SEED_TRAIN workload
                 (keys dets, obs, det_coords), plus whatever graph data the build needs.
        Outputs: none (fills the CAM). NEVER call with SEED_TEST shots.
        """
        raise NotImplementedError

    def lookup(self, key: bytes):
        """Return the cached correction for `key`, or None on a miss."""
        raise NotImplementedError

    def size(self) -> int:
        """Number of CAM entries (for the CAM-size vs hit-rate plot)."""
        raise NotImplementedError

    def evaluate(self, test_shots: dict) -> pd.DataFrame:
        """Run every SEED_TEST shot through the cache path.

        Inputs:  dict from io.load_npz(SHOTS_DIR / f"{tag}.npz") for the SEED_TEST workload.
        Outputs: DataFrame matching contracts.CACHE_COLS, one row per shot
                 (fail_cache is null on a miss).
        Contract: written via io.write_table to TRACES_DIR / f"cache_{tag}.parquet".
        """
        raise NotImplementedError
