# Changelog

## 2.1.0

Evidence for every item marked **(measured)** is in [docs/evidence](docs/evidence/README.md). The `bench/repro` harness
reproduced 13 defects in v2.0.0 (commit 3dcf498); none reproduces in 2.1.0.

### Deliberate changes of meaning

These change numbers that v2.0.0 reported. They are listed so that nobody is surprised; each is a correction.

- **Entries at the first frame.** The first analysed frame has no previous frame, so it has no entry events (v2.0 counted the
  molecules already inside as entries). The initial population is reported separately (`initial_population`).
- **Durations come from time stamps.** v2.0 multiplied frame counts by an assumed step; 2.1 uses the stored times (midpoint
  rule). A single analysed frame has no duration (`n/a`), a format without times is flagged and takes `--dt` / *Frame interval*.
- **Density maps.** The default engine is now the solute-frame engine (solute made whole, solvent wrapped to the solute,
  superposition every frame, bulk measured in the far field). The v2.0 grid is available as `engine="mdanalysis"` and is flagged by the Trust panel.
  A map without superposition is never reported as safe.
- **Frame extraction** refuses inputs that changed since the analysis and never overwrites an earlier extraction (a unique
  prefix is added). Scripts that relied on silent overwriting must pass `--overwrite`.
- **Cancelled runs** return a result marked `complete=False`; it is not exported unless requested, and then into `partial/`.
- **Unknown SOZ parameters** are reported (warning), not silently replaced by defaults.
- **Mean-occupancy standard errors** are the larger of the autocorrelation and five-block estimates (previously autocorrelation only).
- `sozlab-gui` is a new application; the v2.0 GUI (one 11 000-line module, forced X11 platform plugin) is removed. Its project files still open.

### Added

- **Workbench GUI**: one-screen flow (Import, Define, Run, Inspect, Export), live 3D view, live occupancy preview, trust badges and
  panel, definitions on demand, run history with overlay, light/dark themes, reduced motion, drag-and-drop, 1366×768 layout verified.
- **Residence analysis**: censoring flags, complete-visit statistics, Kaplan–Meier, exact S(τ) with origin counts, exponential fit,
  gap tolerance defined in time, stride sensitivity (1×, 2×, 4×), experimental Δt → 0 extrapolation (indicative only).
- **Uncertainty** for occupancy, residence time and rates (autocorrelation and block estimates, Student-*t* intervals).
- **Provenance**: SHA-256 of inputs (sampled above 512 MiB), configuration hash, algorithm versions, environment, `manifest.json`;
  `sozlab-cli verify`; `InputsChangedError` when inputs change mid-analysis.
- **Parallel frame evaluation** (spawned workers, results accumulated in frame order: bitwise identical to serial; starts only when it pays off).
- **Periodic geometry**: triclinic minimum image, whole-molecule COM/COG probes without bond information, cutoff-versus-box checks.
- `sozlab-cli selftest` (synthetic known-answer checks) and `sozlab-cli verify`; `radial` shell mode; data dictionary exported with every run.

### Fixed (reproduced against v2.0.0)

R01 stale density cache · R02 extraction against modified inputs · R03 assumed/uniform timing · R04 initial population counted as entries ·
R05 unflagged window-edge segments · R06 non-unique solvent ids · R07 COM/COG probe of a molecule split by the box · R08 `workers` had no
effect · R09 density ignored the global window · R10 cancelled run exported as complete · R11 silently defaulted parameters ·
R12 cutoff above half the box undetected · R13 GUI could not start off Linux. Details and measurements: `docs/evidence/repro_table.md`.

### Removed

Legacy GUI (the old `src/app/main.py`, `viz_3d.py`), the NGL/WebEngine 3D viewer and its bundled JavaScript, the unused
`water_fallbacks.py`, the tests of the removed GUI, and the dependencies PyQt6-WebEngine, PyOpenGL and scikit-image. The inert H-bond and
water-dynamics result classes remain only so that old imports keep working.

### Known limits

See "What remains" in `docs/evidence/README.md`: residence times from continuous survival depend on the sampling interval; single-run
error bars cannot see variation slower than the run; the 3D view is a software renderer (adequate for ~30 000 solute atoms, not a replacement for VMD);
the experimental extrapolation is unreliable above ~0.5 ps; validation used one force field and one ion.

## 2.0.0

First public release (GUI + CLI).
