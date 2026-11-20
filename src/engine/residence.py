"""Residence-time estimators for solvent in a zone.

What is being measured
----------------------
For one solvent molecule let ``h(k) = 1`` when it is inside the zone at sample ``k``.

* **Segment duration** - how long an observed unbroken visit lasted. Simple and intuitive,
  but biased: visits cut by the window edges are shorter than they really were
  (*censoring*), and the visits you can observe are biased toward short ones when the
  window is short compared with the longest residence.
* **Survival correlation function** (Impey et al. 1983; Rapaport 1983)::

      S(tau) = < h(t0) * prod_{t0<=t<=t0+tau} h(t) > / < h(t0) >

  the probability that a solvent inside the zone at a random time origin has stayed
  inside, without leaving, for the following ``tau``. It uses every time origin, needs no
  event definition, is the quantity other tools report (MDAnalysis ``SurvivalProbability``,
  CPPTRAJ lifetime analyses) and can be integrated to a mean residence time
  ``int S dtau`` or fitted to ``exp(-tau / tau_c)``.

``S`` is computed **exactly and in O(n log n)** from the presence runs rather than from the
per-frame indicator matrix: a run covering samples ``a..b`` contributes ``L - m`` surviving
(origin, lag) pairs at lag ``m`` and ``min(b, n-1-m) - a + 1`` available origins.

Sampling caveat (verified against a Brownian-dynamics ground truth in the validation suite):
any quantity based on *continuous presence* depends on the sampling interval, because a
diffusing molecule re-crosses the zone boundary many times between samples. Discrete
sampling is equivalent to continuous monitoring with the boundary moved outward by
``0.5826 * sqrt(2 D dt)`` (Broadie-Glasserman-Kou continuity correction), so coarser
sampling *lengthens* apparent residence times. SOZLab reports this sensitivity instead of
hiding it (see ``stride_sensitivity`` in the run summary).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz


@dataclass(frozen=True)
class Survival:
    lags: np.ndarray  # sample lags 0..M
    s: np.ndarray  # S(lag), NaN where no origin is available
    origins: np.ndarray  # number of (molecule, origin) pairs contributing at each lag
    n_samples: int


def survival_function(
    starts: Sequence[int],
    ends: Sequence[int],
    n_samples: int,
    max_lag: Optional[int] = None,
) -> Survival:
    """Continuous-presence survival correlation function from inclusive presence runs.

    ``starts[i]..ends[i]`` (inclusive sample indices) are the runs of all molecules pooled.
    """
    a = np.asarray(starts, dtype=np.int64)
    b = np.asarray(ends, dtype=np.int64)
    n = int(n_samples)
    if a.size == 0 or n <= 0:
        lags = np.arange(0, 1)
        return Survival(lags, np.array([np.nan]), np.zeros(1, dtype=np.int64), n)
    length = b - a + 1
    if max_lag is None:
        max_lag = int(length.max())
    max_lag = int(min(max_lag, n - 1))
    lags = np.arange(0, max_lag + 1, dtype=np.int64)

    # numerator: sum over runs with L > m of (L - m)
    ls = np.sort(length)
    cum_l = np.concatenate([[0], np.cumsum(ls)])
    n_le = np.searchsorted(ls, lags, side="right")
    cnt_gt = ls.size - n_le
    num = (cum_l[-1] - cum_l[n_le]) - lags * cnt_gt

    # denominator: sum over runs of (min(b, c) - a + 1)+ with c = n - 1 - m
    c = (n - 1) - lags
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    cum_a = np.concatenate([[0], np.cumsum(a_sorted)])
    cum_b = np.concatenate([[0], np.cumsum(b_sorted)])
    n_a = np.searchsorted(a_sorted, c, side="right")  # runs with a <= c
    n_b = np.searchsorted(b_sorted, c, side="left")  # runs with b < c
    den = ((c + 1) * n_a - cum_a[n_a]) - (c * n_b - cum_b[n_b])

    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(den > 0, num / den, np.nan)
    return Survival(lags, s.astype(np.float64), den.astype(np.int64), n)


def survival_from_presence(presence: np.ndarray, max_lag: Optional[int] = None) -> Survival:
    """Reference O(n^2) implementation from a ``(n_samples, n_molecules)`` boolean matrix.

    Used by the tests as an independent check of :func:`survival_function`.
    """
    pres = np.asarray(presence, dtype=bool)
    n, m = pres.shape
    if max_lag is None:
        max_lag = n - 1
    s = np.full(max_lag + 1, np.nan)
    origins = np.zeros(max_lag + 1, dtype=np.int64)
    for lag in range(max_lag + 1):
        if lag > n - 1:
            break
        window = pres[: n - lag]
        stay = window.copy()
        for j in range(1, lag + 1):
            stay &= pres[j : n - lag + j]
        d = int(window.sum())
        origins[lag] = d
        if d:
            s[lag] = stay.sum() / d
    return Survival(np.arange(max_lag + 1), s, origins, n)


def fill_gaps(presence: np.ndarray, tolerance: int) -> np.ndarray:
    """Treat absences of at most ``tolerance`` samples between two presences as presence."""
    pres = np.asarray(presence, dtype=bool).copy()
    if tolerance <= 0:
        return pres
    n, m = pres.shape
    for j in range(m):
        col = pres[:, j]
        idx = np.flatnonzero(col)
        if idx.size < 2:
            continue
        gaps = np.diff(idx) - 1
        for k in np.flatnonzero((gaps > 0) & (gaps <= tolerance)):
            col[idx[k] + 1 : idx[k + 1]] = True
    return pres


def integrate_survival(surv: Survival, dt: float, tail: float = 0.02,
                       min_origins: int = 20) -> Dict[str, object]:
    """Mean residence time ``int_0^inf S dtau`` by the trapezoid rule over the observed lags.

    Integration stops at the first lag where S falls below ``tail`` or where fewer than
    ``min_origins`` origins remain. If S has not decayed below ``0.10`` by then, the value
    is a *lower bound* (``converged=False``): the window is too short for the residence.
    """
    ok = np.isfinite(surv.s) & (surv.origins >= min_origins)
    if not ok.any() or not np.isfinite(dt):
        return {"tau": float("nan"), "converged": False, "last_lag": 0, "s_last": float("nan")}
    last = int(np.flatnonzero(ok)[-1])
    below = np.flatnonzero(ok & (surv.s < tail))
    stop = int(below[0]) if below.size else last
    lags = surv.lags[: stop + 1]
    s = np.where(np.isfinite(surv.s[: stop + 1]), surv.s[: stop + 1], 0.0)
    tau = float(_trapezoid(s, lags * dt))
    s_last = float(s[-1])
    return {"tau": tau, "converged": bool(s_last <= 0.10), "last_lag": stop, "s_last": s_last}


def fit_exponential(surv: Survival, dt: float, s_min: float = 0.05, min_origins: int = 50,
                    max_lag_fraction: float = 0.5) -> Dict[str, object]:
    """Fit ``S = exp(-tau/tau_c)`` on the early, well-sampled part of the curve.

    Weighted least squares on ``log S`` through the origin (``S(0) = 1`` by definition),
    weights ``origins * S^2`` (counting noise on ``log S`` scales like ``1/sqrt(origins * S)``).
    ``r2`` measures how mono-exponential the decay is; non-exponential decay (anomalous
    diffusion, several residence populations) shows as low ``r2`` and the value should then
    be read as a characteristic time only.
    """
    nan = {"tau_c": float("nan"), "r2": float("nan"), "n_points": 0, "window_lags": (0, 0)}
    if not np.isfinite(dt):
        return nan
    limit = int((surv.n_samples - 1) * max_lag_fraction)
    mask = (surv.lags >= 1) & (surv.lags <= limit) & np.isfinite(surv.s) & (surv.s >= s_min) & (
        surv.origins >= min_origins)
    # use the contiguous early block only
    if not mask.any():
        return nan
    first_bad = np.flatnonzero(~mask & (surv.lags >= 1))
    if first_bad.size:
        mask &= surv.lags < surv.lags[first_bad[0]]
    if mask.sum() < 3:
        return nan
    x = surv.lags[mask] * dt
    y = np.log(surv.s[mask])
    w = surv.origins[mask] * surv.s[mask] ** 2
    slope = float((w * x * y).sum() / (w * x * x).sum())
    if slope >= 0:
        return nan
    pred = slope * x
    ss_res = float((w * (y - pred) ** 2).sum())
    ybar = float((w * y).sum() / w.sum())
    ss_tot = float((w * (y - ybar) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"tau_c": float(-1.0 / slope), "r2": r2, "n_points": int(mask.sum()),
            "window_lags": (int(surv.lags[mask][0]), int(surv.lags[mask][-1]))}


def kaplan_meier(durations: np.ndarray, observed: np.ndarray) -> Dict[str, object]:
    """Kaplan-Meier estimate for visit durations with right censoring (Kaplan & Meier 1958).

    ``observed[i]`` is True for a visit seen to end, False for a visit still in progress when
    the window closed (its true duration is at least ``durations[i]``). Returns the
    restricted mean (area under the curve up to the longest observation) and median.
    """
    d = np.asarray(durations, dtype=np.float64)
    o = np.asarray(observed, dtype=bool)
    ok = np.isfinite(d) & (d >= 0)
    d, o = d[ok], o[ok]
    if d.size == 0:
        return {"rmst": float("nan"), "median": float("nan"), "n": 0, "n_events": 0, "times": np.empty(0), "s": np.empty(0)}
    order = np.argsort(d, kind="stable")
    d, o = d[order], o[order]
    uniq, inverse = np.unique(d, return_inverse=True)
    events = np.bincount(inverse, weights=o.astype(float), minlength=len(uniq))
    total = np.bincount(inverse, minlength=len(uniq)).astype(float)
    at_risk = d.size - np.concatenate([[0], np.cumsum(total)[:-1]])
    surv = np.cumprod(1.0 - events / at_risk)
    times = np.concatenate([[0.0], uniq])
    s_curve = np.concatenate([[1.0], surv])
    rmst = float(np.sum(s_curve[:-1] * np.diff(times)))
    below = np.flatnonzero(s_curve <= 0.5)
    median = float(times[below[0]]) if below.size else float("nan")
    return {"rmst": rmst, "median": median, "n": int(d.size), "n_events": int(o.sum()),
            "times": times, "s": s_curve}


def continuity_corrected_radius(radius: float, diffusion: float, dt: float) -> float:
    """Effective absorbing radius seen by a process sampled every ``dt`` (outward shift)."""
    beta = 0.5825971579390106
    return float(radius + beta * np.sqrt(2.0 * diffusion * dt))
