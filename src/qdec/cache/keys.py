# OWNER: P2
"""Translation-invariant cache keys for clusters, and the isolation check that makes a hit safe."""
from qdec.cache.splitter import Cluster


def canonical_key(cluster: Cluster) -> bytes:
    """Map a cluster to a key that is identical for every translate of the same defect pattern.

    Inputs:  Cluster from splitter.find_clusters.
    Outputs: bytes key for the CAM (e.g. coords shifted to origin, sorted, packed).
    Contract: none on disk; used by cam.ClusterCache.
    Owner:   P2
    """
    raise NotImplementedError


def is_isolated(cluster: Cluster, all_clusters: list[Cluster], boundary) -> bool:
    """True iff the cached correction for `cluster` can be applied without interacting with any
    other cluster or the code boundary (otherwise the shot must go to BP).

    Inputs:  the cluster, every cluster of the same shot, a boundary description of the patch.
    Outputs: bool. A shot is a cache hit only if ALL its clusters hit AND pass this check.
    Contract: feeds CACHE_COLS.cache_hit.
    Owner:   P2
    """
    raise NotImplementedError
