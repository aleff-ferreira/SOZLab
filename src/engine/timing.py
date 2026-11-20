"""Frame timing model.

SOZLab quantities such as residence times, rates and gap tolerances are defined in
*time*, but trajectories deliver *frames*. Version 2.0.0 converted between the two with a
single number, ``dt = median(diff(times))``, and fell back to ``dt = 1.0`` when only one
frame was analysed. That is wrong when

* sampling is irregular (restarts, concatenated runs, variable output interval),
* only one frame is analysed (no interval exists),
* the file format stores no time stamps (MDAnalysis then silently invents
  ``time = frame * 1 ps``).

:class:`FrameTiming` records what is actually known about the analysed frames, and
:func:`segment_boundaries` converts "samples ``a..e`` were all inside the zone" into a
duration using the *midpoint rule*: each sample stands for the interval halfway to its
neighbours. With uniform spacing this reduces exactly to ``(e - a + 1) * dt`` (so
v2.0.0 numbers are reproduced), and with irregular spacing it remains unbiased.

All times are in picoseconds (MDAnalysis base unit).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: Relative tolerance on interval variation below which sampling counts as uniform.
UNIFORM_RTOL = 1e-3

SOURCE_TRAJECTORY = "trajectory"  # time stamps stored in the trajectory file
SOURCE_USER = "user"  # explicit ``analysis.dt_override``
SOURCE_ASSUMED = "assumed"  # format stores no time: MDAnalysis invented frame * 1 ps
SOURCE_MEMORY = "memory"  # in-memory coordinates with a caller-supplied dt


@dataclass(frozen=True)
class FrameTiming:
    """What is known about the time axis of the analysed frames."""

    frames: np.ndarray
    times: np.ndarray
    source: str
    time_unit: str = "ps"
    dt_median: Optional[float] = None
    dt_min: Optional[float] = None
    dt_max: Optional[float] = None
    uniform: bool = False
    monotonic: bool = True
    n_duplicate_times: int = 0
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def n(self) -> int:
        return int(len(self.times))

    @property
    def usable(self) -> bool:
        """True when at least two strictly increasing, finite time stamps exist."""
        return self.n >= 2 and self.monotonic and self.dt_median is not None and self.dt_median > 0

    @property
    def trusted(self) -> bool:
        """True when physical durations can be reported without a caveat."""
        return self.usable and self.source != SOURCE_ASSUMED

    @property
    def dt(self) -> float:
        """Median sample interval in ps, or NaN when no interval exists."""
        return float(self.dt_median) if self.dt_median is not None else float("nan")

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "time_unit": self.time_unit,
            "n_samples": self.n,
            "dt_median": self.dt_median,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "uniform": self.uniform,
            "monotonic": self.monotonic,
            "n_duplicate_times": self.n_duplicate_times,
            "trusted": self.trusted,
            "first_time": float(self.times[0]) if self.n else None,
            "last_time": float(self.times[-1]) if self.n else None,
            "notes": list(self.notes),
        }


def assess_timing(
    frames: Sequence[int],
    times: Sequence[float],
    source: str = SOURCE_TRAJECTORY,
    time_unit: str = "ps",
) -> FrameTiming:
    """Characterise the time axis of the analysed samples."""
    frames_a = np.asarray(frames, dtype=np.int64)
    times_a = np.asarray(times, dtype=np.float64)
    notes: List[str] = []
    n = len(times_a)
    finite = bool(np.all(np.isfinite(times_a)))
    if not finite:
        notes.append("Non-finite time stamps were found; durations are unavailable.")
    diffs = np.diff(times_a) if n > 1 else np.empty(0)
    monotonic = bool(finite and (diffs > 0).all()) if n > 1 else finite
    n_dup = int((diffs == 0).sum()) if n > 1 else 0
    dt_median = dt_min = dt_max = None
    uniform = False
    positive = diffs[np.isfinite(diffs) & (diffs > 0)] if n > 1 else np.empty(0)
    if positive.size:
        dt_median = float(np.median(positive))
        dt_min = float(positive.min())
        dt_max = float(positive.max())
        # Trajectory files such as XTC store time as float32, so intervals jitter by the float32
        # resolution at the largest time (0.0078 ps at t = 1e5 ps). Allow that jitter.
        jitter = 4.0 * float(np.spacing(np.float32(np.max(np.abs(times_a)))))
        uniform = bool(monotonic and (dt_max - dt_min) <= max(UNIFORM_RTOL * dt_median, jitter))
    if n == 1:
        notes.append("Only one frame was analysed; no sampling interval exists, so durations and rates in time units are unavailable.")
    if n > 1 and not monotonic:
        notes.append(
            "Time stamps are not strictly increasing (restart, concatenation or duplicates); "
            "durations in time units are approximate."
        )
    if n > 1 and monotonic and not uniform:
        notes.append(
            f"Sampling is irregular (interval {dt_min:.6g}-{dt_max:.6g} {time_unit}); "
            "durations use the midpoint rule and survival curves assume the median interval."
        )
    if source == SOURCE_ASSUMED:
        notes.append(
            "The trajectory format stores no time stamps; times are frame index x 1 ps assumed by MDAnalysis. "
            "Set analysis.dt_override (ps per trajectory frame) to give physical durations."
        )
    return FrameTiming(
        frames=frames_a,
        times=times_a,
        source=source,
        time_unit=time_unit,
        dt_median=dt_median,
        dt_min=dt_min,
        dt_max=dt_max,
        uniform=uniform,
        monotonic=monotonic,
        n_duplicate_times=n_dup,
        notes=tuple(notes),
    )


def detect_time_source(universe) -> str:
    """Return how the trajectory's time axis was obtained.

    MDAnalysis never fails when a format has no time information: it emits the warning
    "Reader has no dt information, set to 1.0 ps" and reports ``frame * 1 ps``.
    Readers that do store times place ``'time'`` in ``ts.data``.
    """
    traj = getattr(universe, "trajectory", None)
    if traj is None:
        return SOURCE_ASSUMED
    cls_name = type(traj).__name__
    if cls_name == "MemoryReader":
        return SOURCE_MEMORY
    try:
        n_frames = len(traj)
    except Exception:
        n_frames = 0
    if n_frames == 0:
        return SOURCE_ASSUMED
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            ts = traj[0]
            _ = ts.time
        except Exception:
            return SOURCE_ASSUMED
    if any("no dt information" in str(w.message) for w in caught):
        return SOURCE_ASSUMED
    data = getattr(ts, "data", {}) or {}
    if "time" in data:
        return SOURCE_TRAJECTORY
    # Readers such as DCD derive time from header fields rather than ts.data.
    if cls_name in {"DCDReader", "TRJReader", "NCDFReader", "H5MDReader", "TRRReader", "XTCReader", "LAMMPSDCDReader"}:
        return SOURCE_TRAJECTORY
    return SOURCE_ASSUMED


def segment_boundaries(times: Sequence[float]) -> np.ndarray:
    """Interval edges for the midpoint rule, length ``n + 1``.

    Sample ``k`` represents ``[b[k], b[k+1]]``. Interior edges lie halfway between
    neighbouring samples; the two outer edges extend half an interval beyond the first and
    last sample. Requires ``n >= 2``; for ``n == 1`` the result is all-NaN.
    """
    t = np.asarray(times, dtype=np.float64)
    n = len(t)
    b = np.full(n + 1, np.nan)
    if n < 2:
        return b
    b[1:n] = 0.5 * (t[:-1] + t[1:])
    b[0] = t[0] - 0.5 * (t[1] - t[0])
    b[n] = t[-1] + 0.5 * (t[-1] - t[-2])
    return b


def segment_durations(starts: Sequence[int], ends: Sequence[int], times: Sequence[float]) -> np.ndarray:
    """Duration of presence runs given inclusive sample-index ranges ``[start, end]``."""
    b = segment_boundaries(times)
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    return b[e + 1] - b[s]


def gap_tolerance_in_frames(tolerance_time: float, timing: FrameTiming) -> int:
    """Largest number of consecutive absent samples that a tolerance of ``tolerance_time`` bridges.

    ``g`` consecutive absent samples correspond to an absence of ``g * dt``. Following the
    "sustained residence" idea (Impey et al. 1983) an absence is ignored when it does not
    exceed ``t*``, so ``g = floor(t* / dt)``. Expressing the tolerance in time keeps the
    definition independent of the frame stride.
    """
    if not timing.usable or tolerance_time is None or tolerance_time <= 0:
        return 0
    return int(np.floor(tolerance_time / timing.dt_median + 1e-9))
