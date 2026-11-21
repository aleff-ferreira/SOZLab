"""Statistical uncertainty for quantities computed from correlated trajectory samples.

Consecutive MD frames are not independent, so ``std / sqrt(N)`` understates the error of a
time average. Two standard remedies are provided (Flyvbjerg & Petersen 1989; Grossfield
et al., LiveCoMS 1, 5067, 2018):

* :func:`mean_with_error` - integrated autocorrelation time with Sokal's automatic
  windowing gives the statistical inefficiency ``g`` and the effective sample size
  ``N / g`` (used for time-averaged occupancy);
* :func:`block_summary` - contiguous blocks treated as independent replicas (used for
  residence times and rates that are not simple means).
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

try:  # scipy is a dependency of MDAnalysis
    from scipy import stats as _sps
except Exception:  # pragma: no cover
    _sps = None


def _t_crit(dof: float, level: float = 0.95) -> float:
    if _sps is not None and dof >= 1:
        return float(_sps.t.ppf(0.5 + level / 2.0, dof))
    return 1.96


def integrated_autocorr_time(x: Sequence[float], c: float = 5.0) -> Tuple[float, int]:
    """Statistical inefficiency ``g = 1 + 2 * sum_k rho(k)`` and the window used.

    Sokal's automatic windowing: the sum is truncated at the first lag ``M`` with
    ``M >= c * g(M)``, which keeps the noisy tail of the autocorrelation out of the estimate.
    """
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    if n < 4:
        return 1.0, 0
    arr = arr - arr.mean()
    var = float(arr @ arr)
    if var <= 0 or not np.isfinite(var):
        return 1.0, 0
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(arr, n=nfft)
    acf = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
    acf = acf / acf[0]
    taus = 2.0 * np.cumsum(acf) - 1.0
    within = np.arange(n) < c * taus
    if within.all():
        m = n - 1
    else:
        m = int(np.argmin(within))
    return float(max(taus[m], 1.0)), m


def mean_with_error(x: Sequence[float], n_blocks: int = 5) -> Dict[str, float]:
    """Mean with a conservative, autocorrelation-aware standard error and 95% interval.

    Two estimates of the standard error of a time average are computed and the larger is reported:

    * the integrated autocorrelation time (Sokal's automatic windowing), exact for a stationary process whose
      correlations decay inside the run;
    * contiguous blocks treated as independent replicas, which also sees slow variation that the windowed
      autocorrelation misses when a small slow component hides behind a fast one.

    On 4 x 2 ns of TIP3P/NaCl (docs/evidence) the autocorrelation estimate alone covered the long-run mean in only 75% of
    0.25 ns windows; taking the larger of the two raised that to 91% without changing it for long runs.
    """
    arr = np.asarray(x, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"value": float("nan"), "se": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"),
                "n": 0, "n_eff": 0.0, "tau_int": float("nan")}
    mean = float(arr.mean())
    if n < 4:
        return {"value": mean, "se": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"),
                "n": n, "n_eff": float(n), "tau_int": 1.0}
    g, _ = integrated_autocorr_time(arr)
    n_eff = n / g
    se_acf = float(arr.std(ddof=1) * np.sqrt(g / n))
    se, dof, method = se_acf, max(n_eff - 1.0, 1.0), "autocorrelation"
    se_blk = float("nan")
    if n >= 10 * n_blocks:
        means = np.array([arr[a:b].mean() for a, b in block_ranges(n, n_blocks)])
        se_blk = float(means.std(ddof=1) / np.sqrt(len(means)))
        if se_blk > se_acf:
            se, dof, method = se_blk, float(len(means) - 1), "blocks"
    half = _t_crit(dof) * se
    return {"value": mean, "se": se, "ci95_low": mean - half, "ci95_high": mean + half,
            "n": n, "n_eff": float(n_eff), "tau_int": float(g), "se_autocorrelation": se_acf, "se_blocks": se_blk,
            "se_method": method}


def block_ranges(n: int, n_blocks: int) -> Sequence[Tuple[int, int]]:
    """Contiguous ``[start, stop)`` sample ranges of (near-)equal length."""
    n_blocks = max(1, min(int(n_blocks), n))
    edges = np.linspace(0, n, n_blocks + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def block_summary(values: Sequence[float]) -> Dict[str, float]:
    """Mean, standard error and 95% Student-t interval over independent block estimates."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    k = int(arr.size)
    if k == 0:
        return {"value": float("nan"), "se": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n_blocks": 0}
    mean = float(arr.mean())
    if k < 2:
        return {"value": mean, "se": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n_blocks": k}
    se = float(arr.std(ddof=1) / np.sqrt(k))
    half = _t_crit(k - 1) * se
    return {"value": mean, "se": se, "ci95_low": mean - half, "ci95_high": mean + half, "n_blocks": k}
