"""When can each reported quantity be trusted?

Every item is a plain-language statement tied to a measurable condition of the run, so the
GUI, the report and the exported ``qc`` block can show *why* a number should (not) be used.
Levels: ``info`` (context), ``caution`` (use with the stated caveat), ``warning`` (do not use
the affected quantity without fixing the cause).
"""
from __future__ import annotations

import math
from typing import Dict, List

from engine.timing import SOURCE_ASSUMED, FrameTiming


def _fin(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def build_trust(soz_results: Dict[str, object], density_results: Dict[str, object], timing: FrameTiming,
                qc: Dict[str, object]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []

    def add(level: str, ident: str, scope: str, message: str) -> None:
        items.append({"level": level, "id": ident, "scope": scope, "message": message})

    # ---- time axis ----------------------------------------------------------------------------
    if timing.n < 2:
        add("warning", "timing.single_frame", "run",
            "Only one frame was analysed: durations, rates and residence times in time units do not exist.")
    elif timing.source == SOURCE_ASSUMED:
        add("warning", "timing.assumed", "run",
            "The trajectory stores no time stamps; time = frame x 1 ps is an assumption. Set analysis.dt_override "
            "(ps per trajectory frame) before quoting durations or rates.")
    elif not timing.monotonic:
        add("caution", "timing.nonmonotonic", "run",
            "Time stamps are not strictly increasing (restart/concatenation): durations spanning the break are approximate.")
    elif not timing.uniform:
        add("caution", "timing.irregular", "run",
            "Sampling is irregular: visit durations use the midpoint rule, but the survival curve assumes the median interval.")

    # ---- box / periodicity ------------------------------------------------------------------------
    if not qc.get("pbc_used", True):
        add("warning", "pbc.missing", "run", "No valid box vectors: distances are not periodic; solvent near the box edge is mishandled.")
    elif qc.get("pbc_missing_frames", 0):
        add("caution", "pbc.partial", "run", f"{qc['pbc_missing_frames']} frames have no valid box.")

    # ---- per zone ------------------------------------------------------------------------------------
    for name, res in soz_results.items():
        s = res.summary
        scope = f"soz:{name}"
        unc = s.get("uncertainty", {})
        n_eff = (unc.get("mean_n_solvent") or {}).get("n_eff")
        if _fin(n_eff) and n_eff < 30:
            add("caution", "occupancy.few_independent_samples", scope,
                f"The occupancy time series holds only ~{n_eff:.0f} independent samples; the standard error is large.")
        if s.get("occupancy_fraction", 0) < 1e-9:
            add("warning", "occupancy.zero", scope, "The zone was never occupied: check selections, cut-offs and logic.")
            continue
        r = s.get("residence", {}).get("continuous", {})
        nv, nc = r.get("n_visits", 0), r.get("n_visits_complete", 0)
        if nv and _fin(r.get("fraction_censored")) and r["fraction_censored"] > 0.2:
            add("caution", "residence.censored", scope,
                f"{100 * r['fraction_censored']:.0f}% of visits are cut by the window edges; the mean visit duration is biased low "
                "(use the survival-function time or the Kaplan-Meier estimate).")
        if nv and nc < 10:
            add("caution", "residence.few_complete_visits", scope,
                f"Only {nc} complete visits were observed; residence-time estimates are statistically weak.")
        if nv and _fin(r.get("tau_survival_integral")) and not r.get("tau_survival_integral_converged", True):
            add("caution", "residence.not_converged", scope,
                "The survival function has not decayed (S > 0.1 at the longest lag): the window is shorter than the "
                "residence time, so the reported time is a lower bound.")
        for row in (res.stride_sensitivity or [])[1:]:
            m = row["stride_multiplier"]
            ch = row.get("tau_survival_integral_rel_change")
            if _fin(ch) and abs(ch) > 0.10:
                add("caution", "residence.stride_sensitive", scope,
                    f"The residence time changes by {100 * ch:+.0f}% with {m}x coarser sampling: the sampling interval is not "
                    "short compared with the zone's boundary crossings, so the value depends on the stride. Quote it with its interval, or set a "
                    "brief-exit tolerance (conventionally ~2 ps for water, Impey et al. 1983), which keeps it stable while the stride is "
                    "shorter than the tolerance.")
                break
        for row in (res.stride_sensitivity or [])[1:]:
            ch = row.get("entry_rate_per_time_rel_change")
            if _fin(ch) and abs(ch) > 0.15:
                add("caution", "entries.stride_sensitive", scope,
                    f"The entry rate changes by {100 * ch:+.0f}% with {row['stride_multiplier']}x coarser sampling "
                    "(boundary re-crossings between frames are missed). Compare entry rates only at the same stride.")
                break
        ex = s.get("tau_extrapolation") or {}
        if ex.get("valid"):
            add("info", "residence.dt0_estimate", scope,
                f"Experimental continuous-time residence estimate (sqrt(dt) extrapolation): {ex['tau_dt0']:.3g} ps. "
                f"{ex['caveat']}.")
        ch_n = [row.get("mean_n_solvent_rel_change") for row in (res.stride_sensitivity or [])[1:]]
        if any(_fin(c) and abs(c) > 0.10 for c in ch_n):
            add("caution", "occupancy.stride_sensitive", scope,
                "Mean occupancy changed by >10% under coarser sampling: the run is short or strongly correlated.")

    # ---- density maps -------------------------------------------------------------------------------------
    for name, d in density_results.items():
        m = d.metadata
        scope = f"density:{name}"
        if m.get("engine") == "mdanalysis":
            add("caution", "density.legacy_engine", scope,
                "Computed with the legacy MDAnalysis grid: no periodic images, no uncertainty, bulk reference is a single "
                "box-average. Use engine='solute' for publication maps.")
        if not m.get("is_scientifically_safe", True):
            add("warning", "density.unaligned", scope,
                "The map was accumulated without superposition on the solute; solute motion smears it.")
        frac_in = m.get("fraction_of_species_inside_grid")
        if _fin(frac_in) and frac_in < 0.02:
            add("info", "density.small_region", scope,
                f"Only {100 * frac_in:.1f}% of the species' atoms fall inside the grid (region around the solute); this is "
                "expected for a hydration region in a large box.")
        rel = m.get("reliable_fraction")
        if _fin(rel) and rel < 0.15:
            add("caution", "density.low_reliability", scope,
                f"Only {100 * rel:.0f}% of voxels have a relative standard error below 35%: use more frames, a larger "
                "grid spacing or a different region.")
        b = m.get("bulk", {}) or {}
        if b.get("method") == "unavailable":
            add("warning", "density.no_bulk", scope, "No bulk reference density could be determined: g(r) and -ln g are unavailable.")
        ratio = b.get("far_vs_box_ratio")
        if _fin(ratio) and abs(ratio - 1.0) > 0.05:
            add("info", "density.bulk_estimates_differ", scope,
                f"Far-field and box-average bulk densities differ by {100 * (ratio - 1):+.1f}% (the box average includes "
                "the volume excluded by the solute).")
        if _fin(m.get("solute_rotation_max_deg")) and m.get("solute_rotation_max_deg", 0) > 90:
            add("info", "density.large_rotation", scope, "The solute rotated by more than 90 degrees; superposition removes this exactly.")
        cache = m.get("cache", {})
        if cache.get("hit"):
            add("info", "density.cache", scope, "Result reused from the cache (inputs and configuration verified identical by content).")
    return items
