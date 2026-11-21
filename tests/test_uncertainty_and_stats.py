"""Uncertainty estimators against analytic truths; accumulator semantics against the legacy run-length code."""
from __future__ import annotations

import numpy as np
import pytest

from engine import uncertainty as unc
from engine.solvent import SolventRecord
from engine.stats import StatsAccumulator, compute_residence_lengths


def _ar1(rng, n, phi, sigma=1.0):
    x = np.empty(n)
    x[0] = rng.normal(scale=sigma / np.sqrt(1 - phi**2))
    eps = rng.normal(scale=sigma, size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


class TestUncertainty:
    @pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
    def test_statistical_inefficiency_of_ar1(self, phi):
        rng = np.random.default_rng(1)
        g_true = (1 + phi) / (1 - phi)
        gs = [unc.integrated_autocorr_time(_ar1(rng, 20000, phi))[0] for _ in range(12)]
        assert np.mean(gs) == pytest.approx(g_true, rel=0.2)

    def test_standard_error_matches_analytic_and_ci_covers(self):
        phi, n, reps = 0.8, 4000, 300
        rng = np.random.default_rng(2)
        sigma_x = 1.0 / np.sqrt(1 - phi**2)
        g_true = (1 + phi) / (1 - phi)
        se_true = sigma_x * np.sqrt(g_true / n)
        covered, ses = 0, []
        for _ in range(reps):
            r = unc.mean_with_error(_ar1(rng, n, phi))
            ses.append(r["se"])
            covered += r["ci95_low"] <= 0.0 <= r["ci95_high"]
        assert np.mean(ses) == pytest.approx(se_true, rel=0.15)
        assert 0.90 <= covered / reps <= 0.99  # nominal 95%
        # the naive iid estimate would be too small by sqrt(g) ~ 3
        naive = np.mean([_ar1(rng, n, phi).std(ddof=1) / np.sqrt(n) for _ in range(50)])
        assert naive < 0.5 * se_true

    def test_degenerate_inputs(self):
        assert unc.mean_with_error([])["n"] == 0
        r = unc.mean_with_error([3.0, 3.0, 3.0, 3.0, 3.0])
        assert r["value"] == 3.0 and r["se"] == 0.0
        assert np.isnan(unc.mean_with_error([1.0, 2.0])["se"])
        assert unc.integrated_autocorr_time(np.ones(100))[0] == 1.0

    def test_block_helpers(self):
        assert unc.block_ranges(10, 3) == [(0, 3), (3, 7), (7, 10)] or sum(b - a for a, b in unc.block_ranges(10, 3)) == 10
        s = unc.block_summary([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s["value"] == 3.0 and s["se"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5))
        assert s["ci95_low"] < 3.0 < s["ci95_high"] and s["n_blocks"] == 5
        assert np.isnan(unc.block_summary([1.0])["se"]) and unc.block_summary([np.nan])["n_blocks"] == 0


def _records(n):
    return {i: SolventRecord(resindex=i, resid=i + 1, resname="SOL", segid="", atom_indices=[i]) for i in range(n)}


class TestAccumulator:
    def test_first_sample_has_no_events_and_initial_population_is_recorded(self):
        acc = StatsAccumulator(_records(3), 0, 1)
        acc.update(0, 0.0, {0, 1}, frame_label=0)
        acc.update(1, 1.0, {1, 2}, frame_label=1)
        out = acc.finalize()
        assert out["per_frame"]["entries"].tolist() == [0, 1] and out["per_frame"]["exits"].tolist() == [0, 1]
        sm = out["summary"]
        assert sm["initial_population"] == 2 and sm["n_entry_events"] == 1 and sm["n_exit_events"] == 1
        assert sm["entry_rate"] == 1.0  # one entry over one interval (v2.0: 3 entries / 2 frames)

    @pytest.mark.parametrize("gap", [0, 1, 2, 3])
    def test_run_lengths_equal_the_legacy_implementation(self, gap):
        rng = np.random.default_rng(gap)
        n_s, n_m = 120, 15
        state = rng.random(n_m) < 0.3
        sets = []
        for _ in range(n_s):
            flip = rng.random(n_m) < 0.25
            state = np.where(flip, ~state, state)
            sets.append({i for i in range(n_m) if state[i]})
        legacy = compute_residence_lengths(sets, gap)
        acc = StatsAccumulator(_records(n_m), gap, 1, store_frame_table=False)
        for k, s in enumerate(sets):
            acc.update(k, float(k), s, frame_label=k)
        out = acc.finalize()
        for kind, mine, theirs in (("cont", out["residence_cont"], legacy.continuous),
                                   ("inter", out["residence_inter"], legacy.intermittent)):
            assert {k: sorted(v) for k, v in mine.items() if v} == {k: sorted(v) for k, v in theirs.items() if v}, kind

    def test_uniform_durations_reproduce_v20_numbers_and_time_in_zone(self):
        acc = StatsAccumulator(_records(2), 0, 1)
        pattern = [{0}, {0}, {0, 1}, set(), {1}, {1}, {1}, set()]
        for k, s in enumerate(pattern):
            acc.update(k, 10.0 * k, s, frame_label=k)
        ps = acc.finalize()["per_solvent"].set_index("resindex")
        assert ps.loc[0, "mean_res_time_cont"] == pytest.approx(30.0) and ps.loc[0, "frames_present"] == 3
        # molecule 1 is inside at sample 2 (one 10 ps visit) and at samples 4-6 (one 30 ps visit)
        assert ps.loc[1, "n_visits"] == 2 and ps.loc[1, "mean_res_time_cont"] == pytest.approx(20.0)
        assert ps.loc[1, "time_in_zone"] == pytest.approx(40.0) and ps.loc[1, "max_res_time_cont"] == pytest.approx(30.0)
        assert ps.loc[0, "present_at_start"] and not ps.loc[1, "present_at_start"]
        assert ps.loc[0, "n_visits_censored"] == 1  # starts at the first sample

    def test_right_censoring_at_the_last_sample_is_flagged(self):
        acc = StatsAccumulator(_records(1), 0, 1)
        for k, s in enumerate([set(), {0}, {0}, {0}]):
            acc.update(k, float(k), s, frame_label=k)
        out = acc.finalize()
        v = out["visits"]
        assert v["censored_right"].tolist() == [True] and v["censored_left"].tolist() == [False]
        assert out["summary"]["residence"]["continuous"]["n_right_censored"] == 1
        assert out["per_solvent"]["present_at_end"].iloc[0]

    def test_unique_id_map_is_used_in_tables(self):
        recs = _records(2)
        acc = StatsAccumulator(recs, 0, 1, id_map={0: "SOL:1:-#0", 1: "SOL:1:-#1"})
        acc.update(0, 0.0, {0, 1}, frame_label=0)
        out = acc.finalize()
        assert out["per_frame"]["solvent_ids"].iloc[0] == "SOL:1:-#0;SOL:1:-#1"
        assert set(out["per_solvent"]["solvent_id"]) == {"SOL:1:-#0", "SOL:1:-#1"}
