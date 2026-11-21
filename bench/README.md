# Benchmarks and reproducible evidence

| script | what it shows | output |
|---|---|---|
| `repro/repro_v2_flaws.py` | 13 defects of v2.0.0 reproduced on the released code and absent in 2.1 (run against either tree with `PYTHONPATH`) | `docs/evidence/data/repro_*.json`, `docs/evidence/repro_table.md` (`validate/make_evidence_tables.py`) |
| `data/make_tip3p_box.py` | TIP3P + 0.25 M NaCl reference trajectories with OpenMM (fixed seeds; SHA-256 sums in `meta.json`) | `bench/out/tip3p*/` (not committed) |
| `validate/tip3p_validation.py` | membership, coordination, S(τ), uncertainty, stride sensitivity, density vs RDF, parallel identity | `docs/evidence/data/tip3p_validation.json` |
| `validate/dt_extrapolation.py` | residence time vs sampling interval on 0.02 ps data; accuracy of the Δt → 0 extrapolation | `docs/evidence/data/dt_extrapolation.json` |
| `validate/mdanalysis_reference.py` | zone membership vs MDAnalysis `around` on the public triclinic `adk_oplsaa` system | `docs/evidence/data/mdanalysis_reference.json` |
| `validate/perf_compare.py` | wall time and peak memory, v2.0.0 vs this tree | `docs/evidence/data/perf_compare.jsonl` |
| `gui/usability_walkthrough.py` | scripted first-use tasks and recovery scenarios on the real widgets, with screenshots | `docs/evidence/data/usability_walkthrough.json`, `docs/evidence/screens/` |

Everything writes to `bench/out/` or `docs/evidence/`; nothing here is needed to use SOZLab. See `docs/evidence/README.md` for the results and `docs/developer_guide.md` for the commands.
