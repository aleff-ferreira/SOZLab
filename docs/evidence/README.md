# SOZLab 2.1: evidence package

What was checked, against what, with what result, and what is still limited. Every number below comes from a script in
`bench/` that can be re-run (last section); raw outputs are in [`data/`](data/). Measured on one Windows 11 machine, Python 3.12.10,
MDAnalysis 2.9.0, numpy 1.26.4, PyQt6 6.7.1.

## Bottom line

**Where SOZLab advances the field**

1. **Residence time is reported as a measurement, not a number.** Visits cut by the window edges are flagged, four estimators
   are shown together, uncertainty is attached, and the dependence on the sampling interval is *measured* (2×, 4× coarser) rather than
   left to the user. On TIP3P water around Na⁺ the continuous residence time rises from 12.2 ps to 22.6 ps as the sampling interval
   goes from 0.02 to 1.28 ps (table below). The literature review that accompanies this work found no peer-reviewed quantification of that dependence on real MD.
2. **Integrity of derived data.** Density caches are keyed by the *content* of the inputs; frame extraction verifies the inputs and every
   frame by fingerprint and refuses a modified trajectory; every export carries SHA-256 sums, a configuration hash and a manifest that
   `verify` re-checks. The review in `literature_verified.md` found no documentation of input-verified extraction or of hashes/provenance in the outputs of
   MDAnalysis, CPPTRAJ, GROMACS, VMD or MDTraj (marked *not documented*, which does not prove absence); content-keyed caching was not part of that comparison.
3. **A GUI that keeps the science in view**: live 3D zone, live occupancy preview while the cutoff is dragged, trust badges on every
   number, definitions one click away, run history with overlay, and identical results to the CLI (tested down to the per-frame solvent identities).

**Where it matches established tools** (same answer, measured): zone membership equals MDAnalysis' periodic `around` on a public triclinic
system (0 differences in 35 807 molecule-frame memberships); the survival function equals the definition and the `waterdynamics` MDAKit
(3×10⁻⁴, an estimator-weighting difference); coordination numbers equal MDAnalysis `InterRDF` integrals (0.04 %); bulk density equals N/V (0.06 %).

**What remains limited**: see [What remains](#what-remains).

## 1. What was wrong in v2.0.0

Thirteen suspected defects were reproduced on the released code (commit `3dcf498`) with analytically known synthetic data or an
independent numpy oracle, then fixed. All 13 reproduce on v2.0.0; none reproduces on v2.1.0 (`bench/repro/repro_v2_flaws.py`).
Three were silent wrong answers of the kind a reader cannot detect: **R01** a stale density cache returned the grid of an earlier trajectory
with the same path; **R02** extraction wrote frames from a replaced trajectory; **R03** durations were frame count × an assumed step
(and a one-frame run was given one). The full table, with the measured evidence per defect, is in [repro_table.md](repro_table.md).

## 2. Validation

### Known answers (synthetic, no SOZLab code in the reference)

`sozlab-cli selftest` (7 checks, about 10 s) and `tests/` (218 tests: 149 engine and CLI, 25 GUI, 44 renderer): scripted occupancy and
events, irregular time, molecules split across the box (atom / COG / COM probes without bonds), an ideal gas with an exact numpy oracle,
parallel = serial, stale-cache and mutated-input refusal, and an overdamped Brownian particle against the analytic survival function of a sphere
including the discrete-monitoring correction (Broadie, Glasserman & Kou 1997).

### Public data: MDAnalysisTests `adk_oplsaa` (adenylate kinase, 47 681 atoms, triclinic cell 80 Å, 60/60/90°)

| zone: water O within r of the protein | memberships compared | differ from MDAnalysis `around` |
|---|---|---|
| 3 Å | 6 497 | **0** |
| 4 Å | 10 646 | **0** |
| 6 Å | 18 664 | **0** |

### Reference simulation: TIP3P + 0.25 M NaCl (OpenMM, Amber14 + Joung–Cheatham ions, fixed seeds)

1 064 waters, 5 Na⁺, 5 Cl⁻, 3.17 nm box, 300 K, 4 replicas × 2 ns, frames every 0.2 ps; first shell of Na⁺ r_c = 3.20 Å (first minimum of g_NaO).
Water number density in the box: 0.03325 Å⁻³ (in the published TIP3P range 0.0329–0.0335).

| check | against | result |
|---|---|---|
| per-frame zone membership | brute-force minimum-image count (300 frames) | **0 differences** |
| mean zone occupancy | 5 × coordination number integrated from MDAnalysis `InterRDF` | 29.262 vs 29.311 (−0.17 %); replica spread 0.019 |
| survival function S(τ), lags 0–100 ps | brute-force definition on MDAnalysis selections | **identical** (max difference 0) |
| | MDAKit `waterdynamics` `SurvivalProbability` | 3.4×10⁻⁴ (it averages per-origin ratios, SOZLab pools origins) |
| density map vs g(r) of the same ion, r ≥ 3.4 Å | `InterRDF` | RMS Δg 0.023 (0.5 Å grid), 0.010 (0.25 Å grid) |
| coordination number from the density map | `InterRDF` | 5.815 / 5.819 vs 5.813 (0.04 % / 0.1 %) |
| first peak of g(r) | `InterRDF` 5.74 | 4.14 (0.5 Å grid), 5.04 (0.25 Å grid): voxel smoothing, expected |
| bulk density measured in the far field of the map | N/V | 0.033234 vs 0.033253 (−0.06 %) |
| parallel (4 workers) vs serial, 10 000 frames | each other | **bitwise identical** tables |
| v2.1.0 vs v2.0.0 per-frame `n_solvent`, 10 000 frames | each other | **identical** |

Pitfall measured here: averaging a density map radially *after* masking unreliable voxels biased g(r) upward (RMS Δg 1.08 at 0.25 Å, where only 3.5 % of
voxels pass the mask); pooled first, it agrees (0.010). The mask is a display feature.

### Uncertainty across independent runs

Sixteen 0.5 ns windows (4 replicas × 4) of the same system; the reported 95 % interval of each window is compared with the long-run mean.

| quantity | empirical spread ÷ reported SE | 95 % interval covers the long-run mean |
|---|---|---|
| mean occupancy | 0.93 | 13 / 16 (81 %) |
| residence time τ (blocks) | 0.73 (conservative) | 16 / 16 |
| entry rate (blocks) | 0.86 (conservative) | 16 / 16 |

Mean occupancy undercovers: the system has variation slower than the window (ion pairing among 5 ions), which no single-run error bar can
see. The first version of the estimator (autocorrelation only) covered 75 % of 0.25 ns windows; taking the larger of autocorrelation and
5-block errors (now the default) raised that to 91 %. Both are honest limits of short single runs, flagged in the Trust panel when few independent samples exist.

### Residence time depends on how often frames are written

Water in the first shell of Na⁺ (r_c = 3.2 Å). Top: one 0.4 ns trajectory written every 0.02 ps (5 ions; SE 0.8–2.6 ps). Bottom: four 2 ns
trajectories (SD across replicas 0.3–0.9 ps).

| Δt (ps) | 0.02 | 0.04 | 0.08 | 0.16 | 0.32 | 0.64 | 1.28 |
|---|---|---|---|---|---|---|---|
| τ continuous (ps) | 12.2 | 12.6 | 13.1 | 14.4 | 16.8 | 19.6 | 22.6 |

| Δt (ps) | 0.2 | 1.0 | 2.0 | 5.0 | 10.0 |
|---|---|---|---|---|---|
| τ continuous (ps) | 16.4 | 21.9 | 24.1 | 26.4 | 27.9 |
| τ with 2 ps brief-exit tolerance (ps) | 27.3 | 27.6 | 27.8 | 26.4 | 27.9 |
| entry rate (ps⁻¹) | 2.65 | 1.48 | 1.24 | 1.03 | 0.89 |

The tolerance estimate is stable while Δt ≤ t*, which is why Impey et al. introduced t*; Laage & Hynes (2008) caution that t* = 2 ps can
overestimate residence times for low-barrier exchange, so SOZLab reports both and measures the sensitivity instead of choosing for you.
The published mean lifetime of Na⁺ in a 1 M NaCl/optimised-TIP3P box, 39.9 ± 0.4 ps (Mamatkulov & Schwierz 2018), uses different ion
parameters and an undocumented re-entry rule, so it is a plausibility check on the order of magnitude, not a validation of the absolute value.

**The experimental Δt → 0 extrapolation does not hold up on real MD.** Against the directly measured τ(0.02 ps) = 12.2 ps it was within −24…+4 % when
started from Δt ≤ 0.32 ps, and over-estimated by +28 % and +64 % from Δt = 0.64 and 1.28 ps (typical production intervals are longer). It remains in the
Sampling tab, labelled *experimental, indicative only*, with these numbers in its caveat; it is not offered as a result.

## 3. Performance (10 000 frames, 3 202 atoms, serial unless stated)

| | v2.0.0 | v2.1.0 |
|---|---|---|
| SOZ analysis, equal work | 4.19 s, 179 MB peak | 4.21 s, 225 MB |
| SOZ analysis, default options (adds 2×/4× stride sensitivity, uncertainty) | 4.2–4.6 s | 5.0–5.8 s |
| SOZ analysis, 4 worker processes (forced) | — (`workers` had no effect) | 3.7 s (identical output) |
| density map, each version's default engine | 39.2 s, 542 MB | 6.4 s, 219 MB |

The two density engines compute different things (v2.0 aligns the whole trajectory in memory; v2.1 superposes frame by frame in the solute frame), so the row compares cost, not agreement.
The parallel path starts only when the estimated serial time exceeds a threshold; on short runs it stays serial because processes would cost more than they save.

## 4. Comparison with other tools

From `literature_verified.md` (official documentation only; *N/D* = not documented in the pages read, not proof of absence).

| capability | SOZLab 2.1 | MDAnalysis | CPPTRAJ | GROMACS | VMD | MDTraj |
|---|---|---|---|---|---|---|
| per-frame solvent-in-zone counts, boolean zone logic | yes, with the GUI | selections (`around`, …); counting is user code | `watershell` (two shells) | `gmx select` | partly (Tcl) | `compute_neighbors` |
| residence with gap tolerance, per molecule | yes (time-based tolerance, per-visit table) | `SurvivalProbability(intermittency)` (now an MDAKit) | `lifetime fuzz` (frames) | `-olt` (no tolerance) | N/D | N/D |
| time-origin-averaged S(τ) | yes (pooled, with origin counts) | yes | N/D | N/D | N/D | N/D |
| censoring-aware visit statistics, Kaplan–Meier | yes | N/D | N/D | N/D | N/D | N/D |
| uncertainty on reported numbers | yes (autocorrelation + blocks) | N/D | N/D | N/D | partial (per-voxel SD) | N/D |
| periodic distances incl. triclinic | yes (verified on a triclinic public system) | yes | partly | yes | partly | partly |
| density normalised to a measured bulk | yes (far-field vs box) | literature constants | `refdens` constant | partly | N/D | N/D |
| frame extraction by occupancy rule | yes, input-verified | N/D | `filter` + `outtraj` (partly) | `trjconv -drop` (partly) | N/D | N/D |
| hashes / provenance in outputs | yes | N/D | N/D | N/D | N/D | N/D |
| density in the solute frame (wrap + superpose) | yes, built in | user pipeline | `rmsfit` option | `gmx spatial -fit` | N/D | N/D |

GIST (CPPTRAJ) adds solvation energy and entropy terms that SOZLab does not compute; VMD and MDAnalysis are far more general toolkits.
SOZLab is narrower: occupancy, residence and density of solvent around a solute, with the checks above.

## 5. Usability verification

**This is not a user study.** No human took part. A scripted harness (`bench/gui/usability_walkthrough.py`) drives the real widgets with real
input events at 1366×768 and fails when a control is not visible, not labelled, or occluded.

| task of the first-use path | user actions | result |
|---|---|---|
| T1 import a topology and trajectory (drop) | 1 | summary: atoms, frames, box, Δt, solvent detected |
| T2 pick what the zone surrounds, change its distance | 1 click + 3 keys | 3D view and occupancy preview update |
| T3 find, change, and reset an advanced setting | 2 clicks + 1 edit + 2 scrolls | change marked, reset works |
| T4 run | 1 click | complete |
| T5 inspect: read τ ± SE with unit, open its definition, read caveats | 7 clicks | all reachable |
| T6 change a setting, see "settings changed", run again, overlay the earlier run | 3 clicks + 3 keys | overlay shown |
| T7 export data + report, verify against the manifest | 3 clicks | verified |
| R1 selection matches nothing | 1 click to recover | Run blocked with the reason |
| R2 wrong solvent name | 1 click (a "Use SOL" button) | recovered |
| R3 unreadable file | none needed | message names the file and the likely cause |

All 10 pass; 40 interactive controls audited, 13 icon-only controls, all with tooltips; every statistic tile shows its unit; no clipped labels
detected. 25 pytest-qt tests cover the same paths, cancellation (clean cancel and labelled partial result), a stale-results marker, the GUI/CLI
agreement, and layout regressions (no clipped left or right rail in any state, no overlapping tiles at 1080×660 … 1920×1080). 24 screenshots (dark and light, 1100×680, 1366×768,
1920×1080, error states) are in [screens/](screens/): e.g. [results at 1366×768](screens/20_results_1366x768_dark.png),
[light theme](screens/21_results_light_light.png), [trust panel](screens/10_trust_dark.png), [error recovery](screens/13_error_bad_selection_dark.png).

Self-assessment against Nielsen's heuristics (findings, not a score): visibility of status (sticky Run bar with progress, status line, outcome that persists until you change something) ·
match with the real world (MD vocabulary, units everywhere, definitions one click away) · user control (Cancel, Reset all, nothing written until you export) · consistency (one token set for
widgets, charts and 3D) · error prevention (validation before Run, stale-results marker) · recognition over recall (detected parts as chips, defaults) ·
flexibility (Advanced regions, CLI, shortcuts) · minimalist design (one screen, trust shown only where it applies) · error recovery (R1–R3) · help (inline definitions).
**Not done / known gaps:** no undo/redo of setting changes; no screen-reader semantics for the custom-painted stepper and 3D view; no localisation;
no study with first-time researchers; the window's minimum size is 1080×660, and below about 1300×760 the results pane scrolls and the tiles wrap.

## What remains

- **Residence times from continuous survival depend on Δt** (measured above); SOZLab quantifies it but cannot remove it. The tolerance-based estimate depends on t*.
  The Δt → 0 extrapolation is unreliable above ~0.5 ps.
- **Single-run error bars cannot see variation slower than the run** (occupancy intervals covered 81 % of the time in 0.5 ns windows). Longer runs or independent replicas are the only remedy.
- **Validation breadth**: one force-field family (TIP3P, Joung–Cheatham ions), one ion species, 5 ions, a 3.2 nm box, one machine; one public protein system (membership only). Not validated: non-aqueous solvents, polarizable models, very large systems (> 10⁶ atoms), long production runs.
- **Bulk reference** for density needs a far field: a box smaller than ~2 × (region + 8 Å) leaves none and falls back to the box average (flagged).
- **The 3D view is a software renderer** (numpy + QPainter, ~30 000 solute atoms comfortably, adaptive quality); it is not a replacement for VMD, has flat-depth spheres and no translucency.
- **Platforms**: developed and tested on Windows 11. The engine and CLI have no platform-specific code, but Linux and macOS were not run for this release.
- **Not implemented**: hydrogen-bond-resolved residence, solvation thermodynamics (GIST-type), non-spherical zone shapes other than logic combinations of spheres.
- **Release hygiene found along the way**: the `v2.0.0` tag points at commit `6e78c81`, not the released `3dcf498`; `CITATION.cff` carries a placeholder release date.

## Reproduce

See [developer guide](../developer_guide.md#reproducing-the-evidence). Short form:

```bash
pip install -e ".[test,validation]"
python bench/repro/repro_v2_flaws.py --label fixed --out bench/out/repro_fixed.json
python bench/data/make_tip3p_box.py --out bench/out/tip3p --replicas 4 --ns 2.0 --dt-out 0.2
python bench/data/make_tip3p_box.py --out bench/out/tip3p_fine --replicas 1 --ns 0.4 --dt-out 0.02
python bench/validate/tip3p_validation.py && python bench/validate/dt_extrapolation.py
python bench/validate/mdanalysis_reference.py
python bench/gui/usability_walkthrough.py
```

## Sources of the methods

Verified against registries (Crossref, PubMed) in [literature_verified.md](literature_verified.md): Impey, Madden & McDonald 1983 (tolerance time);
Laage & Hynes 2008, J. Phys. Chem. B 112:7697 (sensitivity to t*); Mamatkulov & Schwierz 2018, J. Chem. Phys. 148:074504 (Na⁺ lifetime in TIP3P);
Kaplan & Meier 1958; Flyvbjerg & Petersen 1989; Grossfield et al. 2018, LiveCoMS 1:5067 (uncertainty practice); Kabsch 1976; Theobald 2005;
Nguyen et al. 2012 and Ramsey et al. 2016 (GIST); Michaud-Agrawal et al. 2011 (MDAnalysis); Roe & Cheatham 2013 (CPPTRAJ); McGibbon et al. 2015 (MDTraj).
Added in this session and verified against Crossref: Broadie, Glasserman & Kou 1997, *Math. Finance* 7(4):325–349, doi:10.1111/1467-9965.00035 (continuity correction for discretely monitored
barriers, the √Δt dependence of τ(Δt)); Sokal 1997, *Functional Integration* (NATO ASI B 361), pp. 131–192, doi:10.1007/978-1-4899-0319-8_6 (automatic windowing).
