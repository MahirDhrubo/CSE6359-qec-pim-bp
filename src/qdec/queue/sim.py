# OWNER: P3
"""Discrete-event simulation of a single decoder fed one decode window every round period."""
from dataclasses import dataclass

import numpy as np


@dataclass
class SimResult:
    """Output of simulate(). Owner: P3."""
    latency_ns: np.ndarray    # float[N] per-window latency = queueing wait + service
    t_ns: np.ndarray          # float[M] sample times for queue_depth
    queue_depth: np.ndarray   # int[M] windows waiting or in service at t_ns
    stable: bool              # False if the backlog grows without bound (utilization >= 1)


def simulate(service_ns: np.ndarray, round_period_ns: float, controller=None) -> SimResult:
    """Feed windows arriving every `round_period_ns` through a FIFO decoder.

    Inputs:  service_ns float[N] = SERVICE_COLS.latency_ns for one (tag, system) from
             COSTS_DIR/service_<tag>.parquet (in shot order);
             controller: optional BacklogController; when given, each window's service time is
             re-priced with controller.reprice(row, controller.budget(depth), iter_costs).
    Outputs: SimResult.
    Owner:   P3
    """
    raise NotImplementedError


def max_sustainable_rate(service_ns: np.ndarray, controller=None, tol: float = 0.01) -> float:
    """Highest round rate (rounds per second) at which simulate() stays backlog-stable.

    Inputs:  as simulate(); tol = relative tolerance of the search over round_period_ns.
    Outputs: rate in Hz.
    Owner:   P3
    """
    raise NotImplementedError
