# OWNER: P3
import numpy as np
import pytest

from qdec.queue.sim import max_sustainable_rate, simulate
from qdec.queue.synthetic import same_mean_lognormal

T = 1000.0


def naive(service, period):
    """Event-by-event reference: latency and depth seen at each arrival."""
    finish, latency, depth = [], [], []
    f = 0.0
    for i, s in enumerate(service):
        a = i * period
        f = max(a, f) + s
        finish.append(f)
        latency.append(f - a)
        depth.append(sum(fj > a for fj in finish))
    return np.array(latency), np.array(depth)


def test_constant_service_below_period_never_queues():
    r = simulate(np.full(1000, 0.6 * T), T)
    assert r.queue_depth.max() <= 1
    np.testing.assert_allclose(r.latency_ns, 0.6 * T)
    assert r.stable


def test_service_equal_to_period_is_critically_stable():
    r = simulate(np.full(1000, T), T)
    assert r.queue_depth.max() <= 1 and r.stable


def test_constant_service_above_period_grows_linearly():
    s, n = 1.5 * T, 3000
    r = simulate(np.full(n, s), T)
    i = np.arange(n)
    np.testing.assert_allclose(r.latency_ns, s * (i + 1) - i * T)   # wait grows by s - T per window
    assert np.all(np.diff(r.queue_depth) >= 0)
    slope = np.polyfit(i, r.queue_depth, 1)[0]
    assert slope == pytest.approx(1 - T / s, rel=1e-2)               # 1/3 window per round
    assert not r.stable


def test_matches_event_by_event_reference():
    s = np.random.default_rng(3).lognormal(np.log(800), 1.0, 600)
    r = simulate(s, T)
    lat, depth = naive(s, T)
    np.testing.assert_allclose(r.latency_ns, lat, rtol=1e-12)
    np.testing.assert_array_equal(r.queue_depth, depth)
    np.testing.assert_allclose(r.t_ns, np.arange(len(s)) * T)


def test_stable_flag_follows_utilization():
    df = same_mean_lognormal(0.9 * T, {"heavy": 1.5}, 20_000, seed=0)
    assert simulate(df["latency_ns"].to_numpy(), T).stable
    assert not simulate(df["latency_ns"].to_numpy() * 1.2, T).stable


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        simulate(np.array([]), T)
    with pytest.raises(ValueError):
        simulate(np.ones(5), 0.0)
    with pytest.raises(NotImplementedError):
        simulate(np.ones(5), T, controller=object())


def test_same_mean_lognormal_has_equal_means_and_different_tails():
    df = same_mean_lognormal(900.0, {"light": 0.3, "heavy": 1.5}, 20_000, seed=0)
    g = df.groupby("system")["latency_ns"]
    np.testing.assert_allclose(g.mean(), 900.0)
    assert g.quantile(0.999)["heavy"] > 5 * g.quantile(0.999)["light"]
    again = same_mean_lognormal(900.0, {"light": 0.3, "heavy": 1.5}, 20_000, seed=0)
    np.testing.assert_array_equal(df["latency_ns"], again["latency_ns"])


def test_heavy_tail_builds_deeper_backlog_at_same_mean():
    df = same_mean_lognormal(0.9 * T, {"light": 0.3, "heavy": 1.5}, 20_000, seed=0)
    peak = {k: simulate(g.sort_values("shot_id")["latency_ns"].to_numpy(), T).queue_depth.max()
            for k, g in df.groupby("system")}
    assert peak["light"] <= 10
    assert peak["heavy"] > 10 * peak["light"]


# ---------------------------------------------------------------- max_sustainable_rate


def test_rate_for_constant_service_is_one_over_service():
    assert max_sustainable_rate(np.full(1000, 500.0), tol=1e-3) == pytest.approx(2e6, rel=2e-3)


def test_stability_rate_is_one_over_mean_service():
    s = np.random.default_rng(1).lognormal(np.log(400), 1.0, 20_000)
    assert max_sustainable_rate(s, tol=1e-3) == pytest.approx(1e9 / s.mean(), rel=2e-3)


def test_latency_budget_constant_service():
    s = np.full(1000, 500.0)
    assert max_sustainable_rate(s, latency_budget_ns=500.0, tol=1e-3) == pytest.approx(2e6, rel=2e-3)
    assert max_sustainable_rate(s, latency_budget_ns=499.0) == 0.0     # one decode alone too slow


def test_latency_budget_penalizes_heavy_tail_at_same_mean():
    df = same_mean_lognormal(500.0, {"light": 0.3, "heavy": 1.5}, 20_000, seed=0)
    s = {k: g.sort_values("shot_id")["latency_ns"].to_numpy() for k, g in df.groupby("system")}
    stab = {k: max_sustainable_rate(v) for k, v in s.items()}
    assert stab["light"] == pytest.approx(stab["heavy"], rel=0.02)    # same mean -> same rate
    tail = {k: max_sustainable_rate(v, latency_budget_ns=10_000.0) for k, v in s.items()}
    assert tail["light"] > 1.5 * tail["heavy"]
    assert tail["heavy"] < stab["heavy"]


def test_tighter_budget_never_raises_rate():
    s = np.random.default_rng(2).lognormal(np.log(400), 1.0, 5000)
    rates = [max_sustainable_rate(s, latency_budget_ns=b) for b in (20_000, 10_000, 5_000)]
    assert rates[0] >= rates[1] >= rates[2]


def test_rate_rejects_all_zero_service():
    with pytest.raises(ValueError):
        max_sustainable_rate(np.zeros(10))
