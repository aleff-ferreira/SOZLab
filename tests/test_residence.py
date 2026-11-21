"""Survival correlation function, estimators and censoring-aware statistics."""
from __future__ import annotations

import numpy as np
import pytest

from engine import residence as res
from engine.synthetic import sphere_survival, BGK_BETA


def _runs_from_presence(presence):
    starts, ends = [], []
    n, m = presence.shape
    for j in range(m):
        col = presence[:, j]
        i = 0
        while i < n:
            if col[i]:
                k = i
                while k + 1 < n and col[k + 1]:
                    k += 1
                starts.append(i); ends.append(k)
                i = k + 1
            else:
                i += 1
    return np.array(starts), np.array(ends)


@pytest.mark.parametrize("seed,n,m,p", [(0, 60, 8, 0.5), (1, 200, 30, 0.2), (2, 37, 5, 0.85), (3, 120, 40, 0.05)])
def test_survival_matches_bruteforce_reference(seed, n, m, p):
    rng = np.random.default_rng(seed)
    # persistent random presence so runs are non-trivial
    pres = np.zeros((n, m), bool)
    state = rng.random(m) < p
    for t in range(n):
        flip = rng.random(m) < 0.15
        state = np.where(flip, rng.random(m) < p, state)
        pres[t] = state
    a, b = _runs_from_presence(pres)
    got = res.survival_function(a, b, n, max_lag=n - 1)
    ref = res.survival_from_presence(pres, max_lag=n - 1)
    np.testing.assert_array_equal(got.origins[: len(ref.origins)], ref.origins)
    np.testing.assert_allclose(got.s, ref.s, equal_nan=True, rtol=0, atol=1e-12)


def test_survival_simple_known_values():
    # one molecule present in samples 2..5 of 10 -> S(m) = (4-m)/(...)
    a, b = np.array([2]), np.array([5])
    s = res.survival_function(a, b, 10)
    assert s.s[0] == 1.0
    # lag 1: origins 2..5 (4 of them, all have a window of 1 within 10 samples); survivors k=2,3,4 -> 3/4
    assert s.s[1] == pytest.approx(3 / 4)
    assert s.s[3] == pytest.approx(1 / 4)


def test_edge_truncation_lowers_origin_count_not_survival():
    # a run touching the last sample cannot be followed for long lags: those origins are excluded
    a, b = np.array([6]), np.array([9])
    s = res.survival_function(a, b, 10)
    assert s.origins[0] == 4 and s.origins[3] == 1  # origins k<=6 only
    assert s.s[3] == pytest.approx(1.0)  # the one origin (k=6) did survive 3 samples


def test_fill_gaps_matches_intermittent_definition():
    pres = np.array([[1], [0], [1], [0], [0], [1], [1]], bool)
    assert fill_list(res.fill_gaps(pres, 1)) == [1, 1, 1, 0, 0, 1, 1]
    assert fill_list(res.fill_gaps(pres, 2)) == [1, 1, 1, 1, 1, 1, 1]
    assert fill_list(res.fill_gaps(pres, 0)) == [1, 0, 1, 0, 0, 1, 1]


def fill_list(a):
    return [int(x) for x in a[:, 0]]


def test_exponential_decay_recovered_by_fit_and_integral():
    rng = np.random.default_rng(5)
    tau_true, dt, n, m = 8.0, 1.0, 4000, 60
    # two-state Markov presence with exit rate 1/tau, strong re-entry so the pool is large
    p_exit = 1 - np.exp(-dt / tau_true)
    pres = np.zeros((n, m), bool)
    state = rng.random(m) < 0.5
    for t in range(n):
        leave = state & (rng.random(m) < p_exit)
        enter = (~state) & (rng.random(m) < 0.05)
        state = (state & ~leave) | enter
        pres[t] = state
    a, b = _runs_from_presence(pres)
    surv = res.survival_function(a, b, n, max_lag=200)
    fit = res.fit_exponential(surv, dt)
    integ = res.integrate_survival(surv, dt)
    # discrete-time geometric survival S(m) = (1-p)^m: the trapezoid integral is dt/p - dt/2 = tau + O(dt^2/tau)
    assert fit["tau_c"] == pytest.approx(tau_true, rel=0.06)
    assert fit["r2"] > 0.99
    assert integ["converged"] and integ["tau"] == pytest.approx(tau_true, rel=0.05)


def test_kaplan_meier_handles_censoring():
    # exponential(tau=10) lifetimes truncated by a window of 15: naive mean is biased low
    rng = np.random.default_rng(7)
    life = rng.exponential(10.0, 4000)
    window = 15.0
    obs = life <= window
    dur = np.minimum(life, window)
    naive = dur[obs].mean()  # complete segments only: strongly biased low
    km = res.kaplan_meier(dur, obs)
    # restricted mean up to 15 of exp(10) = 10*(1-exp(-1.5)) = 7.77
    assert km["rmst"] == pytest.approx(10 * (1 - np.exp(-1.5)), rel=0.04)
    assert naive < 6.5 < km["rmst"]
    assert km["n_events"] == int(obs.sum())


def test_kaplan_meier_no_censoring_equals_empirical_mean():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    km = res.kaplan_meier(d, np.ones(4, bool))
    assert km["rmst"] == pytest.approx(d.mean() - 0.0, rel=1e-12) or km["rmst"] == pytest.approx(2.5)


def test_sphere_survival_ground_truth_properties():
    t = np.linspace(0, 40, 2001)
    s = sphere_survival(t, radius=4.0, diffusion=0.25)
    assert s[0] == pytest.approx(1.0, abs=2e-3)
    assert np.trapz(s, t) == pytest.approx(4.0**2 / (15 * 0.25), rel=2e-3)  # R^2/(15 D)
    assert BGK_BETA == pytest.approx(0.5826, abs=1e-4)
