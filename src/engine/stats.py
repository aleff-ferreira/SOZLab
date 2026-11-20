"""Occupancy, event and residence statistics for SOZ results.

Everything per molecule is derived from *presence runs* ``(a, b)`` (inclusive sample indices
of an unbroken stay in the zone). That makes the per-frame cost proportional to the number
of molecules that *change* state, not to the zone size, and lets durations, censoring and
the survival correlation function all come from the same primitive.

Definitions (see docs/science/definitions.md)
---------------------------------------------
* ``n_solvent(k)``       number of solvent molecules in the zone at sample ``k``.
* **entry / exit**       a molecule absent at sample ``k-1`` and present at ``k`` (or the
  reverse). Sample 0 has no predecessor, so it has no entries or exits: v2.0.0 counted the
  whole initial population as entries.
* **visit** (continuous residence) one run; **intermittent visit**: runs separated by at most
  ``gap_tolerance`` absent samples are merged (Impey et al. 1983, "sustained residence").
* **censoring**          a run starting at sample 0 is *left-censored*, one ending at the last
  sample is *right-censored*: their true duration is unknown, so they are flagged and kept
  out of the "complete" averages.
* **duration**           midpoint rule on the actual time stamps (see :mod:`engine.timing`).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from engine import residence as res_mod
from engine.solvent import SolventRecord
from engine.timing import FrameTiming, assess_timing, SOURCE_TRAJECTORY, segment_boundaries
from engine.uncertainty import block_ranges, block_summary, mean_with_error


@dataclass
class ResidenceStats:
    continuous: Dict[int, List[float]]
    intermittent: Dict[int, List[float]]


class StatsAccumulator:
    """Streaming accumulator; feed it one ``present`` set per analysed sample."""

    def __init__(
        self,
        solvent_records: Dict[int, SolventRecord],
        gap_tolerance: int,
        frame_stride: int,
        store_ids: bool = True,
        store_frame_table: bool = True,
        id_map: Optional[Dict[int, str]] = None,
    ) -> None:
        self.solvent_records = solvent_records
        self.gap_tolerance = int(gap_tolerance)
        self.frame_stride = int(frame_stride)
        self.store_ids = store_ids
        self.store_frame_table = store_frame_table
        self.id_map = id_map or {idx: rec.stable_id for idx, rec in solvent_records.items()}

        self.frame_times: List[float] = []
        self.frame_labels: List[int] = []
        self.n_solvent: List[int] = []
        self.n_entries: List[int] = []
        self.n_exits: List[int] = []
        self.frame_rows: List[Dict[str, object]] = []

        self.entries_count: Dict[int, int] = defaultdict(int)
        self.exits_count: Dict[int, int] = defaultdict(int)
        self.cont_segments: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self._open_cont: Dict[int, int] = {}
        self.prev_set: Set[int] = set()
        self.initial_set: Set[int] = set()
        self.last_sample_index = -1

    # ------------------------------------------------------------------ streaming
    def _close_run(self, resindex: int, start: int, end: int) -> None:
        self.cont_segments[resindex].append((start, end))

    def update(
        self,
        sample_index: int,
        time: float,
        present: Set[int],
        frame_label: int | None = None,
        frame_fp: str | None = None,
    ) -> None:
        label = sample_index if frame_label is None else frame_label
        first = len(self.frame_times) == 0
        if first:
            entries: Set[int] = set()
            exits: Set[int] = set()
            self.initial_set = set(present)
            for r in present:
                self._open_cont[r] = sample_index
        else:
            entries = present - self.prev_set
            exits = self.prev_set - present
            for r in exits:
                start = self._open_cont.pop(r)
                self._close_run(r, start, sample_index - 1)
                self.exits_count[r] += 1
            for r in entries:
                self._open_cont[r] = sample_index
                self.entries_count[r] += 1

        if self.store_frame_table:
            ids = ""
            if self.store_ids and present:
                ids = ";".join(sorted(self.id_map[idx] for idx in present))
            row = {
                "frame": label,
                "time": time,
                "n_solvent": len(present),
                "solvent_ids": ids,
                "entries": len(entries),
                "exits": len(exits),
            }
            if frame_fp is not None:
                row["frame_fp"] = frame_fp
            self.frame_rows.append(row)

        self.frame_times.append(float(time))
        self.frame_labels.append(int(label))
        self.n_solvent.append(len(present))
        self.n_entries.append(len(entries))
        self.n_exits.append(len(exits))
        self.prev_set = set(present)
        self.last_sample_index = sample_index

    # ------------------------------------------------------------------ results
    def _all_runs(self, gap_tolerance: Optional[int] = None) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]:
        """Continuous and intermittent runs, including runs still open at the last sample.

        Intermittent runs merge continuous runs separated by at most ``gap_tolerance`` absent
        samples ("sustained residence", Impey et al. 1983).
        """
        g = self.gap_tolerance if gap_tolerance is None else int(gap_tolerance)
        n = len(self.frame_times)
        cont = {r: list(v) for r, v in self.cont_segments.items()}
        for r, start in self._open_cont.items():
            cont.setdefault(r, []).append((start, n - 1))
        inter: Dict[int, List[Tuple[int, int]]] = {}
        for r, segs in cont.items():
            merged = [list(segs[0])]
            for a, b in segs[1:]:
                if a - merged[-1][1] - 1 <= g:
                    merged[-1][1] = b
                else:
                    merged.append([a, b])
            inter[r] = [(a, b) for a, b in merged]
        return cont, inter

    @staticmethod
    def _flatten(runs: Dict[int, List[Tuple[int, int]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        owners, starts, ends = [], [], []
        for r, segs in runs.items():
            for a, b in segs:
                owners.append(r)
                starts.append(a)
                ends.append(b)
        return (np.asarray(owners, dtype=np.int64), np.asarray(starts, dtype=np.int64),
                np.asarray(ends, dtype=np.int64))

    def finalize(
        self,
        time_unit: str = "ps",
        timing: Optional[FrameTiming] = None,
        n_blocks: int = 5,
        gap_tolerance: Optional[int] = None,
    ) -> Dict[str, object]:
        n_frames = len(self.frame_times)
        if n_frames == 0:
            raise ValueError("No frames provided for stats")
        if timing is None:
            timing = assess_timing(self.frame_labels, self.frame_times, SOURCE_TRAJECTORY, time_unit)
        times = np.asarray(self.frame_times, dtype=np.float64)
        dt = timing.dt  # NaN when no interval exists
        bounds = segment_boundaries(times)

        gap = self.gap_tolerance if gap_tolerance is None else int(gap_tolerance)
        cont, inter = self._all_runs(gap)
        c_own, c_a, c_b = self._flatten(cont)
        i_own, i_a, i_b = self._flatten(inter)

        def _durations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            if a.size == 0:
                return np.empty(0)
            if n_frames < 2:
                return np.full(a.size, np.nan)
            if timing.uniform or not timing.monotonic:
                # frame-count x median interval: exact for uniform sampling, best available otherwise
                return (b - a + 1) * dt
            return bounds[b + 1] - bounds[a]

        c_dur = _durations(c_a, c_b)
        i_dur = _durations(i_a, i_b)
        c_left = c_a == 0
        c_right = c_b == n_frames - 1
        i_left = i_a == 0
        i_right = i_b == n_frames - 1

        # --- per-solvent table -------------------------------------------------------------
        by_c: Dict[int, List[int]] = defaultdict(list)
        for idx, owner in enumerate(c_own):
            by_c[int(owner)].append(idx)
        by_i: Dict[int, List[int]] = defaultdict(list)
        for idx, owner in enumerate(i_own):
            by_i[int(owner)].append(idx)

        rows = []
        present_end = set(self.prev_set)
        for resindex, record in self.solvent_records.items():
            ci = by_c.get(resindex, [])
            ii = by_i.get(resindex, [])
            frames_present = int(sum(c_b[k] - c_a[k] + 1 for k in ci))
            c_d = [float(c_dur[k]) for k in ci]
            i_d = [float(i_dur[k]) for k in ii]
            complete = [float(c_dur[k]) for k in ci if not (c_left[k] or c_right[k])]
            first_seen = self.frame_labels[int(min(c_a[k] for k in ci))] if ci else None
            last_seen = self.frame_labels[int(max(c_b[k] for k in ci))] if ci else None
            rows.append(
                {
                    "solvent_id": self.id_map[resindex],
                    "resindex": resindex,
                    "resname": record.resname,
                    "resid": record.resid,
                    "segid": record.segid,
                    "frames_present": frames_present,
                    "occupancy_pct": 100.0 * frames_present / n_frames,
                    "first_seen_frame": first_seen,
                    "last_seen_frame": last_seen,
                    "entries": int(self.entries_count.get(resindex, 0)),
                    "mean_res_time_cont": mean(c_d) if c_d and not np.isnan(c_d).any() else (0.0 if not c_d else float("nan")),
                    "median_res_time_cont": median(c_d) if c_d and not np.isnan(c_d).any() else (0.0 if not c_d else float("nan")),
                    "mean_res_time_inter": mean(i_d) if i_d and not np.isnan(i_d).any() else (0.0 if not i_d else float("nan")),
                    "median_res_time_inter": median(i_d) if i_d and not np.isnan(i_d).any() else (0.0 if not i_d else float("nan")),
                    # --- added in 2.1 (all previous columns are unchanged) ---
                    "present_at_start": resindex in self.initial_set,
                    "present_at_end": resindex in present_end,
                    "n_visits": len(ci),
                    "n_visits_complete": len(complete),
                    "n_visits_censored": len(ci) - len(complete),
                    "mean_res_time_cont_complete": mean(complete) if complete else float("nan"),
                    "max_res_time_cont": max(c_d) if c_d else 0.0,
                    "time_in_zone": float(np.nansum(c_d)) if c_d else 0.0,
                }
            )
        per_solvent_df = pd.DataFrame(rows)
        per_solvent_df.sort_values(by=["occupancy_pct", "solvent_id"], ascending=False, inplace=True)

        # --- global summaries ----------------------------------------------------------------
        n_series = np.asarray(self.n_solvent, dtype=np.float64)
        occupancy_frames = int((n_series > 0).sum())
        n_intervals = n_frames - 1
        total_entries = int(sum(self.n_entries))
        total_exits = int(sum(self.n_exits))
        span = float(times[-1] - times[0]) if n_frames > 1 else float("nan")

        entry_rate = total_entries / n_intervals if n_intervals > 0 else float("nan")
        exit_rate = total_exits / n_intervals if n_intervals > 0 else float("nan")
        rate_ok = timing.usable and np.isfinite(span) and span > 0

        surv_c = res_mod.survival_function(c_a, c_b, n_frames)
        surv_i = res_mod.survival_function(i_a, i_b, n_frames)

        def _residence_block(a, b, dur, left, right, surv) -> Dict[str, object]:
            complete_mask = ~(left | right)
            durations_ok = np.isfinite(dur)
            km = None
            if timing.usable and a.size:
                # left-censored visits have unknown start: excluded; right-censored enter as censored
                use = ~left & durations_ok
                km = res_mod.kaplan_meier(dur[use], ~right[use])
            integ = res_mod.integrate_survival(surv, dt) if timing.usable else {"tau": float("nan"), "converged": False}
            fit = res_mod.fit_exponential(surv, dt) if timing.usable else {"tau_c": float("nan"), "r2": float("nan")}
            block = {
                "n_visits": int(a.size),
                "n_visits_complete": int(complete_mask.sum()),
                "n_left_censored": int(left.sum()),
                "n_right_censored": int(right.sum()),
                "fraction_censored": float((~complete_mask).mean()) if a.size else float("nan"),
                "mean_visit_duration_all": float(np.nanmean(dur)) if a.size and durations_ok.any() else float("nan"),
                "mean_visit_duration_complete": float(np.nanmean(dur[complete_mask & durations_ok]))
                if (complete_mask & durations_ok).any() else float("nan"),
                "max_visit_duration": float(np.nanmax(dur)) if a.size and durations_ok.any() else float("nan"),
                "km_restricted_mean": float(km["rmst"]) if km else float("nan"),
                "km_median": float(km["median"]) if km else float("nan"),
                "tau_survival_integral": float(integ["tau"]),
                "tau_survival_integral_converged": bool(integ["converged"]),
                "tau_survival_fit": float(fit["tau_c"]),
                "tau_survival_fit_r2": float(fit["r2"]),
            }
            return block

        residence = {
            "time_unit": time_unit,
            "continuous": _residence_block(c_a, c_b, c_dur, c_left, c_right, surv_c),
            "intermittent": _residence_block(i_a, i_b, i_dur, i_left, i_right, surv_i),
            "gap_tolerance_samples": gap,
            "gap_tolerance_time": gap * dt if timing.usable else float("nan"),
        }

        # --- survival table ------------------------------------------------------------------
        m = max(len(surv_c.lags), len(surv_i.lags))
        lags = np.arange(m)
        def _pad(arr, fill=np.nan):
            out = np.full(m, fill, dtype=float)
            out[: len(arr)] = arr
            return out
        survival_df = pd.DataFrame(
            {
                "lag_samples": lags,
                "tau": lags * dt if timing.usable else np.full(m, np.nan),
                "S_continuous": _pad(surv_c.s),
                "S_intermittent": _pad(surv_i.s),
                "origins_continuous": _pad(surv_c.origins, 0),
                "origins_intermittent": _pad(surv_i.origins, 0),
            }
        )

        # --- uncertainty -----------------------------------------------------------------------
        uncertainty = self._uncertainty(n_series, c_a, c_b, i_a, i_b, times, timing, n_blocks, dt)

        per_frame_df = pd.DataFrame(self.frame_rows) if self.store_frame_table else pd.DataFrame()
        summary = {
            "n_frames": n_frames,
            "frame_stride": self.frame_stride,
            "dt": dt,
            "time_unit": time_unit,
            "occupancy_fraction": occupancy_frames / n_frames,
            "mean_n_solvent": float(n_series.mean()),
            "median_n_solvent": float(np.median(n_series)),
            "max_n_solvent": int(n_series.max()) if n_series.size else 0,
            "n_solvent_hist": dict(pd.Series(self.n_solvent).value_counts().sort_index()),
            "entry_rate": entry_rate,
            "exit_rate": exit_rate,
            # --- added in 2.1 ---
            "n_intervals": n_intervals,
            "initial_population": len(self.initial_set),
            "n_entry_events": total_entries,
            "n_exit_events": total_exits,
            "entry_rate_per_time": total_entries / span if rate_ok else float("nan"),
            "exit_rate_per_time": total_exits / span if rate_ok else float("nan"),
            "timing": timing.to_dict(),
            "residence": residence,
            "uncertainty": uncertainty,
        }

        visits_df = self._visits_table(cont, c_own, c_a, c_b, c_dur, c_left, c_right, times)
        return {
            "per_frame": per_frame_df,
            "per_solvent": per_solvent_df,
            "summary": summary,
            "residence_cont": {r: [b - a + 1 for a, b in segs] for r, segs in cont.items()},
            "residence_inter": {r: [b - a + 1 for a, b in segs] for r, segs in inter.items()},
            "segments_cont": cont,
            "segments_inter": inter,
            "survival": survival_df,
            "visits": visits_df,
            "timing": timing,
        }

    # ------------------------------------------------------------------ helpers
    def _visits_table(self, cont, owners, a, b, dur, left, right, times) -> pd.DataFrame:
        if a.size == 0:
            return pd.DataFrame(columns=["solvent_id", "resindex", "start_frame", "end_frame", "start_time",
                                         "end_time", "duration", "censored_left", "censored_right"])
        order = np.lexsort((a, owners))
        return pd.DataFrame(
            {
                "solvent_id": [self.id_map[int(o)] for o in owners[order]],
                "resindex": owners[order],
                "start_frame": np.asarray(self.frame_labels)[a[order]],
                "end_frame": np.asarray(self.frame_labels)[b[order]],
                "start_time": times[a[order]],
                "end_time": times[b[order]],
                "duration": dur[order],
                "censored_left": left[order],
                "censored_right": right[order],
            }
        )

    def _uncertainty(self, n_series, c_a, c_b, i_a, i_b, times, timing, n_blocks, dt) -> Dict[str, object]:
        out: Dict[str, object] = {
            "method": {
                "means": "integrated autocorrelation time (Sokal window); se = std*sqrt(g/N)",
                "residence_and_rates": f"{n_blocks} contiguous blocks treated as independent (Student-t 95% interval)",
            }
        }
        out["mean_n_solvent"] = mean_with_error(n_series)
        out["occupancy_fraction"] = mean_with_error((n_series > 0).astype(float))
        n = len(n_series)
        ranges = block_ranges(n, n_blocks)
        if len(ranges) < 2 or n < 20:
            out["blocks_note"] = "too few samples for block estimates"
            return out
        ent = np.asarray(self.n_entries, dtype=float)
        rates, taus, fits, conv = [], [], [], []
        for s0, s1 in ranges:
            span = float(times[s1 - 1] - times[s0]) if s1 - s0 > 1 else float("nan")
            # entries between samples s0+1..s1-1 (first sample of a block has no in-block predecessor)
            rates.append(ent[s0 + 1 : s1].sum() / span if (timing.usable and span > 0) else float("nan"))
            keep = (c_b >= s0) & (c_a < s1)
            a = np.maximum(c_a[keep], s0) - s0
            b = np.minimum(c_b[keep], s1 - 1) - s0
            sv = res_mod.survival_function(a, b, s1 - s0)
            if timing.usable:
                integ = res_mod.integrate_survival(sv, dt, min_origins=10)
                taus.append(integ["tau"])
                conv.append(bool(integ["converged"]))
                fits.append(res_mod.fit_exponential(sv, dt, min_origins=10)["tau_c"])
        out["entry_rate_per_time"] = block_summary(rates)
        if timing.usable:
            out["tau_survival_integral"] = block_summary(taus)
            out["tau_survival_fit"] = block_summary(fits)
            out["block_length_time"] = float(np.mean([times[s1 - 1] - times[s0] for s0, s1 in ranges]))
            out["tau_blocks_converged_fraction"] = float(np.mean(conv)) if conv else float("nan")
        return out


def compute_residence_lengths(
    frame_sets: List[Set[int]],
    gap_tolerance: int,
) -> ResidenceStats:
    """Run lengths (in samples) of continuous and intermittent presence for each molecule."""
    cont_lengths: Dict[int, List[int]] = defaultdict(list)
    inter_lengths: Dict[int, List[int]] = defaultdict(list)

    current_cont: Dict[int, int] = {}
    current_inter: Dict[int, int] = {}
    last_present: Dict[int, int] = {}

    for frame_index, present in enumerate(frame_sets):
        for resindex in list(current_cont.keys()):
            if resindex not in present:
                start = current_cont.pop(resindex)
                cont_lengths[resindex].append(frame_index - start)

        for resindex in present:
            if resindex not in current_cont:
                current_cont[resindex] = frame_index

        for resindex in present:
            if resindex not in current_inter:
                current_inter[resindex] = frame_index
            else:
                if frame_index - last_present[resindex] > gap_tolerance + 1:
                    start = current_inter[resindex]
                    inter_lengths[resindex].append(last_present[resindex] - start + 1)
                    current_inter[resindex] = frame_index
            last_present[resindex] = frame_index

    last_frame = len(frame_sets) - 1
    for resindex, start in current_cont.items():
        cont_lengths[resindex].append(last_frame - start + 1)

    for resindex, start in current_inter.items():
        end = last_present.get(resindex, last_frame)
        inter_lengths[resindex].append(end - start + 1)

    return ResidenceStats(continuous=cont_lengths, intermittent=inter_lengths)


def compute_stats(
    frame_sets: List[Set[int]],
    frame_times: List[float],
    solvent_records: Dict[int, SolventRecord],
    gap_tolerance: int,
    frame_stride: int,
    time_unit: str = "ps",
) -> Dict[str, object]:
    accumulator = StatsAccumulator(
        solvent_records=solvent_records,
        gap_tolerance=gap_tolerance,
        frame_stride=frame_stride,
        store_ids=True,
        store_frame_table=True,
    )
    for sample_index, present in enumerate(frame_sets):
        time = frame_times[sample_index] if sample_index < len(frame_times) else float(sample_index)
        frame_label = sample_index * frame_stride
        accumulator.update(sample_index, time, present, frame_label=frame_label)

    return accumulator.finalize(time_unit=time_unit)
