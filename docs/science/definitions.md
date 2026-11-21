# What SOZLab measures, and when to trust it

Every quantity below is defined from first principles, with the assumption it rests on and the failure it is prone to.
The same text is shown in the Workbench (the **?** buttons), written into every export (`DATA_DICTIONARY.md`) and kept
in one place in the code (`src/engine/dictionary.py`).

## 1. The zone

A solvent molecule is **in the zone** in a frame when its *probe* (by default the water oxygen) lies within a **cutoff**
of any atom of the **selection**, using the minimum-image convention of that frame's box (orthorhombic or triclinic).
Zones combine with AND / OR / NOT. A zone around several atoms is the *union* of spheres, not one sphere around their centre.

Assumptions and limits:

- Minimum image is only unique when the cutoff is below half the shortest box edge. Larger cutoffs are flagged.
- A *centre-of-mass or centre-of-geometry* probe must be computed on a molecule that is whole. SOZLab unwraps each solvent
  molecule before averaging (v2.0 averaged raw, periodically split coordinates, and could place the probe across the box).
- Geometric selections such as `around 5 resname LIG` change from frame to frame; they must be marked *updating*,
  otherwise they are evaluated once on the first frame.
- Multiple "shells" can be chained (shell *k* measured from the molecules of shell *k−1*, the v2.0 behaviour kept for
  compatibility) or radial (every radius measured from the selection). The two differ; the zone description says which.

## 2. Time

Durations come from the trajectory's time stamps, not from an assumed constant step. Each analysed frame stands for the
interval halfway to its neighbours (the **midpoint rule**), so a stay covering *k* evenly spaced frames lasts *k·Δt*. The
sampling is classified as uniform (tolerating float32 jitter), irregular, non-monotonic (restarts), or assumed (no time in the
file). A single frame has **no** duration: v2.0 reported a one-frame residence as if it had a time. Whatever depends on time
(residence, rates) is reported as `n/a` when the time axis cannot be trusted, and the Trust panel says why.

## 3. Occupancy

*n(t)* is the number of molecules in the zone in an analysed frame. **Mean occupancy** is its time average, **occupied
fraction** the share of frames with *n ≥ 1*. Consecutive frames are correlated, so the standard error is not
*σ/√N*. SOZLab computes two estimates and reports the larger: the integrated autocorrelation time with Sokal's automatic
window (exact when correlations decay inside the run) and five contiguous blocks treated as replicas (which also sees slow
variation that hides behind a fast one). Intervals are Student-*t*.

*Check*: on 4 × 2 ns of TIP3P/NaCl, split into 0.25–1 ns windows, the autocorrelation estimate alone covered the long-run
mean in 75 % of 0.25 ns windows; taking the larger of the two raised that to 91 % (see `docs/evidence`). With fewer than ~30 independent samples the Trust panel warns that the error is large.

## 4. Entries and exits

An **entry** is a molecule absent in the previous analysed frame and present now; an exit is the reverse. The first
frame has no predecessor, so it has no events (v2.0 counted the initial population as entries). Events are counted *between
sampled frames*: a molecule that enters and leaves between two frames is invisible. The **entry rate** is therefore a
function of the sampling interval, not a property of the system. On the validation system it falls from 4.6 /ps at
Δt = 0.02 ps to 0.89 /ps at Δt = 10 ps. Compare rates only at equal Δt, and always quote Δt.

## 5. Visits and residence

A **visit** is one unbroken stay of one molecule in the zone. Visits touching the first or last analysed frame are
**censored**: their true duration is unknown, and dropping them or counting them as complete both bias the mean. SOZLab
flags them (`censored_left`, `censored_right`) and reports several estimators side by side because each answers a slightly different question:

| Estimator | Question it answers | Weakness |
|---|---|---|
| mean visit duration (complete visits) | typical length of a stay that was seen from start to end | biased low when stays are comparable to the window |
| Kaplan–Meier restricted mean / median | the same, using the censored visits correctly | needs enough visits |
| **S(τ)** and τ = ∫S dτ | average over all time origins of the chance a molecule present at t₀ is still continuously present at t₀+τ (Impey et al. 1983; the survival probability of MDAnalysis `waterdynamics`) | needs a window ≫ the decay time; the tail is noisy where few origins remain |
| exponential fit of S | the time constant if the decay is single-exponential | wrong when it is not (check R²) |
| intermittent S with a **brief-exit tolerance** t* | residence ignoring excursions shorter than t* (conventionally 2 ps for water) | depends on t*; Laage & Hynes 2008 show it can overestimate for low-barrier exchange |

S(τ) pools every origin whose window fits: S(τ) = Σ N_τ / Σ N₀. This is exact for that definition (verified bitwise against a
brute-force implementation) and differs from the per-origin average of the `waterdynamics` package by ≈3×10⁻⁴ on the
validation data.

### Residence times depend on how often you sample

For a boundary crossed by diffusion, sampling every Δt misses brief excursions, so the **continuous** residence time
grows with Δt: the effective boundary moves outward by β√(2DΔt) with β = 0.5826 (Broadie, Glasserman & Kou 1997), giving
τ(Δt) ≈ τ₀(1 + c√Δt). Measured here on TIP3P water around Na⁺ with a trajectory written every 0.02 ps:

| Δt (ps) | 0.02 | 0.04 | 0.08 | 0.16 | 0.32 | 0.64 | 1.28 |
|---|---|---|---|---|---|---|---|
| τ continuous (ps) | 12.2 | 12.6 | 13.1 | 14.4 | 16.8 | 19.6 | 22.6 |

(one 0.4 ns trajectory, 5 ions; standard error 0.8–2.6 ps). The same quantity from four 2 ns trajectories at Δt = 0.2–10 ps
rises from 16.4 to 27.9 ps, while the intermittent estimate with t* = 2 ps stays at 27.3–27.9 ps for Δt ≤ 2 ps.
I found no peer-reviewed quantification of this dependence on real MD; the table above is a small one.

What to do about it:

- Quote τ **with its Δt**; SOZLab's *Sampling* tab and Trust panel measure the sensitivity (2× and 4× coarser) rather than assume it.
- Or use a brief-exit tolerance t* longer than Δt, which makes the value independent of the stride as long as Δt < t*.
- SOZLab also offers an **experimental** extrapolation to Δt → 0 (straight line in √Δt over 1×, 2×, 4× strides). It recovered the exact
  continuous-time value of a Brownian ground truth to −2…−4 %, but on real MD data it is within −24…+4 % only for Δt ≤ 0.32 ps and
  over-estimates by +28…+64 % at 0.64–1.28 ps. It is **indicative only**; do not quote it.

## 6. Density maps

The map is the time-averaged number density of a solvent *probe atom* in the frame of the solute. Per frame: (1) the solute
is made whole, (2) every solvent molecule is moved to the periodic image nearest the solute, (3) the solute is optimally
superposed (Kabsch) on the reference, (4) probe atoms are binned. Skipping any step smears or splits the density.

- **Bulk reference.** g(r) = ρ(r)/ρ_bulk and −ln g need ρ_bulk. SOZLab measures it in the far field of the map and compares it with the
  box average (the ratio is reported; on the validation system the two agree to 0.06 %). Quoting a literature constant instead can be
  off by 1–2 % relative to the model's own bulk (published TIP3P densities range 0.0329–0.0335 Å⁻³).
- **Uncertainty.** A standard error per voxel comes from blocks of frames; voxels are marked *unreliable* where the relative error exceeds
  35 % or outside the fully covered radius. Masking is for display: **average radially before masking**, because masking first
  keeps the voxels that fluctuated high and biases g upward (documented in `bench/validate`).
- **Resolution.** A 0.5 Å grid smooths a sharp first peak (g = 4.1 against 5.7 from an RDF); a 0.25 Å grid reaches 5.0. The integrated
  coordination number agrees to 0.1 % at either spacing.
- **Cache.** A cached map is reused only when the content of the inputs (not their path), the configuration and the algorithm version match.

## 7. Frame extraction

Frames are chosen from the per-frame table by a rule (`n_solvent>=2`, minimum run length, gap bridging). Before writing, the inputs are
checked by content and each frame by a fingerprint recorded at analysis time (a hash of ~128 spread atoms and the box). A modified,
replaced or truncated trajectory is refused with an explanation. Output files get a unique prefix; an earlier extraction is never overwritten silently.

## 8. Provenance

Every export records the SOZLab version, the algorithm revisions of each quantity, the hash of the full configuration, SHA-256 of the
topology and trajectory (the full file up to 512 MiB, sampled above), the Python environment and the command line, and ends with a
`manifest.json` of SHA-256 sums of everything written. `sozlab-cli verify` re-checks both the files and that the inputs are unchanged.

## 9. Quick reference: when not to trust a number

| Symptom | Affects | Why |
|---|---|---|
| "time stamps assumed" | residence, rates | the file has no times |
| cutoff > half the box | everything | the zone overlaps its own image |
| few independent samples | mean occupancy ± | the error is large |
| > 20 % of visits cut by the window | mean visit duration | biased low; use S(τ) or Kaplan–Meier |
| S(τ) has not decayed | residence | the window is shorter than the residence time: lower bound |
| stride sensitivity > 10 % | residence, entry rate | the value depends on Δt |
| density without superposition | map | solute motion smears it |
| low reliable fraction | map | too few frames or too fine a grid |
