from engine.stats import StatsAccumulator
from engine.solvent import SolventRecord


def test_stats_accumulator_basic():
    records = {
        0: SolventRecord(resindex=0, resid=1, resname="SOL", segid="", atom_indices=[0]),
        1: SolventRecord(resindex=1, resid=2, resname="SOL", segid="", atom_indices=[1]),
    }
    acc = StatsAccumulator(solvent_records=records, gap_tolerance=0, frame_stride=1)
    acc.update(0, 0.0, {0}, frame_label=0)
    acc.update(1, 1.0, {0, 1}, frame_label=1)
    acc.update(2, 2.0, set(), frame_label=2)
    stats = acc.finalize()
    per_solvent = stats["per_solvent"]
    assert per_solvent.loc[per_solvent["resindex"] == 0, "frames_present"].iloc[0] == 2
    assert per_solvent.loc[per_solvent["resindex"] == 1, "frames_present"].iloc[0] == 1
