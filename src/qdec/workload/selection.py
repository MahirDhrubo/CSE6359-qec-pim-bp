# OWNER: P1
"""Explicit sweep cuts without modifying the shared workload grids."""
from dataclasses import replace


def select_workloads(grid, *, code=None, max_distance=None, shots=None):
    """Distance cap applies to surface only; retain gross unless a code is selected.

    A shot cap only reduces counts. Any cut must be reported as a reduced profile,
    never as completed FULL_GRID. Run cuts in a separate working directory.
    """
    if shots is not None and shots < 1:
        raise ValueError("shots must be positive")
    if max_distance is not None and max_distance < 3:
        raise ValueError("max_distance must be >= 3")
    selected = [w for w in grid if (code is None or w.code == code)
                and (max_distance is None or w.code != "surface" or w.distance <= max_distance)]
    if not selected:
        raise ValueError("no workloads match the requested profile")
    return [replace(w, shots=min(w.shots, shots)) if shots is not None else w for w in selected]
