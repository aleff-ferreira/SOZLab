"""Statistics and event calculations for SOZ results."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from engine.solvent import SolventRecord


@dataclass
class ResidenceStats:
    continuous: Dict[int, List[float]]
    intermittent: Dict[int, List[float]]


@dataclass
class StatsAccumulator:
    solvent_records: Dict[int, SolventRecord]
    gap_tolerance: int
    frame_stride: int
    store_ids: bool = True
    store_frame_table: bool = True

    def __post_init__(self) -> None:
        self.frame_times: List[float] = []
        self.n_solvent: List[int] = []
        self.entries_per_frame: List[Set[int]] = []
        self.exits_per_frame: List[Set[int]] = []
        self.frame_rows: List[Dict[str, object]] = []

        self.frames_present_count: Dict[int, int] = defaultdict(int)
        self.first_seen: Dict[int, int] = {}
        self.last_seen: Dict[int, int] = {}
        self.entries_count: Dict[int, int] = defaultdict(int)

        self.current_cont: Dict[int, int] = {}
        self.current_inter: Dict[int, int] = {}
        self.last_present: Dict[int, int] = {}
        self.cont_lengths: Dict[int, List[int]] = defaultdict(list)
        self.inter_lengths: Dict[int, List[int]] = defaultdict(list)
        self.cont_segments: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.prev_set: Set[int] = set()
        self.last_sample_index = -1

    def update(
        self,
        sample_index: int,
        time: float,
        present: Set[int],
        frame_label: int | None = None,
    ) -> None:
        entries = present - self.prev_set
        exits = self.prev_set - present
        label = sample_index if frame_label is None else frame_label

        if self.store_frame_table:
            present_ids = []
            if self.store_ids:
                present_ids = [self.solvent_records[idx].stable_id for idx in sorted(present)]
            self.frame_rows.append(
                {
                    "frame": label,
                    "time": time,
                    "n_solvent": len(present),
                    "solvent_ids": ";".join(present_ids),
                    "entries": len(entries),
                    "exits": len(exits),
                }
            )

        self.frame_times.append(time)
        self.n_solvent.append(len(present))
        self.entries_per_frame.append(entries)
        self.exits_per_frame.append(exits)

        for resindex in present:
            self.frames_present_count[resindex] += 1
            if resindex not in self.first_seen:
                self.first_seen[resindex] = label
            self.last_seen[resindex] = label

        for resindex in entries:
            self.entries_count[resindex] += 1

        # Continuous segments
        for resindex in list(self.current_cont.keys()):
            if resindex not in present:
                start = self.current_cont.pop(resindex)
                end = sample_index - 1
                self.cont_lengths[resindex].append(end - start + 1)
                self.cont_segments[resindex].append((start, end))

        for resindex in present:
            if resindex not in self.current_cont:
                self.current_cont[resindex] = sample_index

        # Intermittent segments
        for resindex in present:
            if resindex not in self.current_inter:
                self.current_inter[resindex] = sample_index
            else:
                if sample_index - self.last_present[resindex] > self.gap_tolerance + 1:
                    start = self.current_inter[resindex]
                    end = self.last_present[resindex]
                    self.inter_lengths[resindex].append(end - start + 1)
                    self.current_inter[resindex] = sample_index
            self.last_present[resindex] = sample_index

        self.prev_set = present
        self.last_sample_index = sample_index

    def finalize(self, time_unit: str = "ps") -> Dict[str, object]:
        n_frames = len(self.frame_times)
        if n_frames == 0:
            raise ValueError("No frames provided for stats")

        if len(self.frame_times) > 1:
            dt = float(np.median(np.diff(self.frame_times)))
        else:
            dt = 1.0

        last_frame = self.last_sample_index if self.last_sample_index >= 0 else n_frames - 1
        for resindex, start in self.current_cont.items():
            end = last_frame
            self.cont_lengths[resindex].append(end - start + 1)
            self.cont_segments[resindex].append((start, end))

        for resindex, start in self.current_inter.items():
            end = self.last_present.get(resindex, last_frame)
            self.inter_lengths[resindex].append(end - start + 1)

        per_solvent_rows = []
        for resindex, record in self.solvent_records.items():
            present_frames = self.frames_present_count.get(resindex, 0)
            occupancy_pct = 100.0 * present_frames / n_frames
            cont_list = self.cont_lengths.get(resindex, [])
            inter_list = self.inter_lengths.get(resindex, [])
            cont_times = [length * dt for length in cont_list]
            inter_times = [length * dt for length in inter_list]
            per_solvent_rows.append(
                {
                    "solvent_id": record.stable_id,
                    "resindex": resindex,
                    "resname": record.resname,
                    "resid": record.resid,
                    "segid": record.segid,
                    "frames_present": present_frames,
                    "occupancy_pct": occupancy_pct,
                    "first_seen_frame": self.first_seen.get(resindex),
                    "last_seen_frame": self.last_seen.get(resindex),
                    "entries": self.entries_count.get(resindex, 0),
                    "mean_res_time_cont": mean(cont_times) if cont_times else 0.0,
                    "median_res_time_cont": median(cont_times) if cont_times else 0.0,
                    "mean_res_time_inter": mean(inter_times) if inter_times else 0.0,
                    "median_res_time_inter": median(inter_times) if inter_times else 0.0,
                }
            )

        per_solvent_df = pd.DataFrame(per_solvent_rows)
        per_solvent_df.sort_values(by=["occupancy_pct", "solvent_id"], ascending=False, inplace=True)

        occupancy_frames = sum(1 for count in self.n_solvent if count > 0)
        occupancy_fraction = occupancy_frames / n_frames

        per_frame_df = pd.DataFrame(self.frame_rows) if self.store_frame_table else pd.DataFrame()

        summary = {
            "n_frames": n_frames,
            "frame_stride": self.frame_stride,
            "dt": dt,
            "time_unit": time_unit,
            "occupancy_fraction": occupancy_fraction,
            "mean_n_solvent": float(np.mean(self.n_solvent)),
            "median_n_solvent": float(np.median(self.n_solvent)),
            "max_n_solvent": int(np.max(self.n_solvent)) if self.n_solvent else 0,
            "n_solvent_hist": dict(pd.Series(self.n_solvent).value_counts().sort_index()),
            "entry_rate": float(sum(len(items) for items in self.entries_per_frame) / n_frames),
            "exit_rate": float(sum(len(items) for items in self.exits_per_frame) / n_frames),
        }

        return {
            "per_frame": per_frame_df,
            "per_solvent": per_solvent_df,
            "summary": summary,
            "residence_cont": self.cont_lengths,
            "residence_inter": self.inter_lengths,
            "entries_per_frame": self.entries_per_frame,
            "exits_per_frame": self.exits_per_frame,
            "segments_cont": self.cont_segments,
        }


def compute_residence_lengths(
    frame_sets: List[Set[int]],
    gap_tolerance: int,
) -> ResidenceStats:
    cont_lengths: Dict[int, List[int]] = defaultdict(list)
    inter_lengths: Dict[int, List[int]] = defaultdict(list)

    current_cont: Dict[int, int] = {}
    current_inter: Dict[int, int] = {}
    last_present: Dict[int, int] = {}

    for frame_index, present in enumerate(frame_sets):
        # Continuous residence segments
        for resindex in list(current_cont.keys()):
            if resindex not in present:
                start = current_cont.pop(resindex)
                cont_lengths[resindex].append(frame_index - start)

        for resindex in present:
            if resindex not in current_cont:
                current_cont[resindex] = frame_index

        # Intermittent residence segments
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


def compute_events(frame_sets: List[Set[int]]) -> Tuple[List[Set[int]], List[Set[int]]]:
    entries_per_frame: List[Set[int]] = []
    exits_per_frame: List[Set[int]] = []

    prev_set: Set[int] = set()
    for present in frame_sets:
        entries = present - prev_set
        exits = prev_set - present
        entries_per_frame.append(entries)
        exits_per_frame.append(exits)
        prev_set = present

    return entries_per_frame, exits_per_frame


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
