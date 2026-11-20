"""Scientific validation against exact / independent references.

Free Brownian solvent around a fixed solute; the zone is a sphere of radius R. For molecules that start uniformly inside
the zone the *continuous-time* survival probability is S(t) = (6/pi^2) sum_n n^-2 exp(-n^2 pi^2 D t / R^2)
(Carslaw & Jaeger) with mean R^2/(15 D). Sampling every dt lets molecules leave and return unseen, which is equivalent
to moving the absorbing wall outward by 0.5826 sqrt(2 D dt) (Broadie-Glasserman-Kou continuity correction) while
the start stays uniform in R: coarser sampling *lengthens* apparent residence times. Three layers are checked:

1. the engine reproduces an independent brute-force evaluation of the same definition on the stored coordinates;
2. the discrete survival curve agrees with the analytic discrete-monitoring theory (and is far from continuous theory);
3. the residence time, entry rate and their stride dependence follow the theory SOZLab reports on.
"""
from __future__ import annotations

import numpy as np
import pytest

mda = pytest.importorskip("MDAnalysis")

from _helpers import make_project
from engine import residence as res
from engine import synthetic as syn
from engine.analysis import SOZAnalysisEngine
from engine.models import AnalysisOptions

R, D, DT, L = 4.0, 0.25, 0.5, 36.0


def _predicted_integral(radius_dt, cut=0.02, step=DT):
    """Integral of the theoretical S on the same lag grid and with the same truncation rule as the engine."""
    t = np.arange(0, 400) * step
    s = syn.sphere_survival_discrete(t, R, D, radius_dt)
    stop = int(np.flatnonzero(s < cut)[0])
    return float(np.trapz(s[: stop + 1], t[: stop + 1]))


N_REP, T_REP = 4, 3000


@pytest.fixture(scope="module")
def brownian_runs(tmp_path_factory):
    """Four independent engine runs (different seeds): pooling them makes single-run noise (~5% in tau) small."""
    runs = []
    for k in range(N_REP):
        tmp = tmp_path_factory.mktemp(f"brownian{k}")
        system = syn.make_brownian_system(n_waters=800, n_frames=T_REP, box=L, diffusion=D, dt=DT, seed=21 + k)
        gro, xtc = system.write(tmp, "b")
        project = make_project(gro, xtc, tmp / "out", cutoff=R, analysis=AnalysisOptions(
            store_min_distances=False, workers=1, store_ids=False, uncertainty_blocks=5))
        runs.append((SOZAnalysisEngine(project).run(), gro, xtc))
    return runs


def test_mean_occupancy_equals_density_times_zone_volume(brownian_runs):
    expected = 800 * (4 / 3) * np.pi * R**3 / L**3
    means = [r.soz_results["Z"].summary["mean_n_solvent"] for r, _, _ in brownian_runs]
    se = np.mean([r.soz_results["Z"].summary["uncertainty"]["mean_n_solvent"]["se"] for r, _, _ in brownian_runs]) / np.sqrt(N_REP)
    assert abs(np.mean(means) - expected) < 4 * se


def test_engine_survival_equals_independent_bruteforce_on_the_stored_coordinates(brownian_runs):
    """Layer 1: same definition, independent code path (numpy float64 distances, O(n^2) survival)."""
    result, gro, xtc = brownian_runs[0]
    u = mda.Universe(str(gro), str(xtc))
    box = np.array(u.dimensions[:3], dtype=np.float64)
    presence = []
    for ts in u.trajectory:
        d = u.atoms.positions[1:].astype(np.float64) - u.atoms.positions[0]
        d -= np.round(d / box) * box
        presence.append(np.linalg.norm(d, axis=1) <= R)
    ref = res.survival_from_presence(np.array(presence), max_lag=12)
    got = result.soz_results["Z"].survival
    np.testing.assert_allclose(got["S_continuous"].to_numpy()[:13], ref.s, atol=2e-3)
    np.testing.assert_array_equal(got["origins_continuous"].to_numpy()[:13][:3], ref.origins[:3])


def test_survival_curve_matches_discrete_monitoring_theory_not_continuous_theory(brownian_runs):
    """Layer 2: pooled over independent runs; the leading-order (continuity-corrected) theory itself is good to ~0.02."""
    curves = [r.soz_results["Z"].survival for r, _, _ in brownian_runs]
    lags = np.arange(1, 17)
    measured = np.mean([c["S_continuous"].to_numpy()[lags] for c in curves], axis=0)
    tau = lags * DT
    discrete = syn.sphere_survival_discrete(tau, R, D, DT)
    continuous = syn.sphere_survival(tau, R, D)
    assert np.abs(measured - discrete).max() < 0.03
    assert np.abs(measured - continuous).max() > 0.08  # ignoring the sampling interval is clearly wrong
    # first lag from first principles: exit probability per step ~ (3 sigma / R) / sqrt(2 pi), sigma = sqrt(2 D dt)
    sigma = np.sqrt(2 * D * DT)
    assert measured[0] == pytest.approx(1 - (3 * sigma / R) / np.sqrt(2 * np.pi), abs=0.03)


def test_residence_time_and_its_stride_dependence_follow_theory(brownian_runs):
    """Layer 3."""
    taus = []
    for m in (1, 2, 4):
        vals = []
        for r, _, _ in brownian_runs:
            rows = {x["stride_multiplier"]: x for x in r.soz_results["Z"].stride_sensitivity}
            assert set(rows) == {1, 2, 4}
            vals.append(rows[m]["tau_survival_integral"])
        pred = _predicted_integral(DT * m, step=DT * m)
        got = float(np.mean(vals))
        assert got == pytest.approx(pred, rel=0.10), f"stride x{m}: {got:.3f} vs {pred:.3f}"
        taus.append(got)
    assert taus[0] < taus[1] < taus[2]  # coarser sampling -> longer apparent residence
    assert taus[0] > 1.2 * (R**2 / (15 * D))  # even dt = 0.5 ps overstates the continuous-time value by > 20%
    assert any("residence.stride_sensitive" in {t["id"] for t in r.qc_summary["trust"]} for r, _, _ in brownian_runs)


def test_entry_rate_depends_on_the_sampling_interval(brownian_runs):
    for r, _, _ in brownian_runs:
        rows = {x["stride_multiplier"]: x for x in r.soz_results["Z"].stride_sensitivity}
        # boundary re-crossings between frames are missed: the rate per unit time falls with coarser sampling
        assert rows[2]["entry_rate_per_time"] < rows[1]["entry_rate_per_time"] * 0.95
        assert rows[4]["entry_rate_per_time"] < rows[2]["entry_rate_per_time"] * 0.95
        for m in (2, 4):  # occupancy, by contrast, is stride invariant
            assert rows[m]["mean_n_solvent"] == pytest.approx(rows[1]["mean_n_solvent"], rel=0.05)
    assert any(t["id"] == "entries.stride_sensitive" for t in brownian_runs[0][0].qc_summary["trust"])


def test_block_uncertainty_is_reported_with_five_blocks(brownian_runs):
    for r, _, _ in brownian_runs:
        u = r.soz_results["Z"].summary["uncertainty"]["tau_survival_integral"]
        assert u["n_blocks"] == 5 and u["se"] > 0 and u["ci95_low"] < u["value"] < u["ci95_high"]


def test_dt_to_zero_extrapolation_recovers_the_continuous_time_residence_time(brownian_runs):
    """Candidate estimator, measured against the exact answer R^2/(15 D): within 8% (raw sampling error is > 25%)."""
    est = []
    for r, _, _ in brownian_runs:
        ex = r.soz_results["Z"].summary["tau_extrapolation"]
        assert ex["valid"] and ex["status"] == "experimental" and ex["n_points"] == 3
        est.append(ex["tau_dt0"])
    truth = R**2 / (15 * D)
    assert np.mean(est) == pytest.approx(truth, rel=0.08)
    raw = np.mean([r.soz_results["Z"].stride_sensitivity[0]["tau_survival_integral"] for r, _, _ in brownian_runs])
    assert abs(np.mean(est) - truth) < 0.5 * abs(raw - truth)
