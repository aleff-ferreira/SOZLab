import pytest
import numpy as np
import pandas as pd
import MDAnalysis as mda
from engine.analysis import _merge_hydration_results, HydrationResult

def test_merge_hydration_results_empty():
    """Test merging empty results returns correct structure."""
    with pytest.raises(ValueError, match="Cannot merge empty result list"):
        _merge_hydration_results([])

def test_merge_hydration_results_basic():
    """Test merging two partial results."""
    # Mock HydrationResults
    # We need to construct them manually as they are usually output from _run_hbond_hydration
    
    # Result 1: Frame 0, Residue 1
    df1 = pd.DataFrame({
        "resindex": [0], "resid": [1], "resname": ["ALA"], "segid": ["A"],
        "freq_total": [1.0], "freq_given_soz": [1.0], "soz_populated_freq": [1.0],
        "frames_with_contact_total": [1], 
        "frames_with_contact_given_soz": [1]
    })
    r1 = HydrationResult(
        name="test",
        table=df1,
        frame_times=[0.0],
        frame_labels=[0],
        contact_frames_total={0: [0]},
        contact_frames_given_soz={0: [0]},
        mode="soz"
    )
    
    # Result 2: Frame 1, Residue 2 (different residue)
    df2 = pd.DataFrame({
        "resindex": [1], "resid": [2], "resname": ["GLY"], "segid": ["A"],
        "freq_total": [1.0], "freq_given_soz": [1.0], "soz_populated_freq": [1.0],
        "frames_with_contact_total": [1], 
        "frames_with_contact_given_soz": [1]
    })
    r2 = HydrationResult(
        name="test",
        table=df2,
        frame_times=[1.0],
        frame_labels=[1],
        contact_frames_total={1: [1]},
        contact_frames_given_soz={1: [1]},
        mode="soz"
    )
    
    # Total frames = 2
    merged = _merge_hydration_results([r1, r2])
    
    assert len(merged.table) == 2
    assert set(merged.table["resname"]) == {"ALA", "GLY"}
    # Freq total should be 0.5 for each since they appear in 1 out of 2 frames
    row_ala = merged.table[merged.table["resname"] == "ALA"].iloc[0]
    assert row_ala["freq_total"] == 0.5
    
    row_gly = merged.table[merged.table["resname"] == "GLY"].iloc[0]
    assert row_gly["freq_total"] == 0.5
