
import numpy as np
import pandas as pd
import MDAnalysis as mda
from typing import Optional

def compute_wor_fallback(u: mda.Universe, selection: str, options) -> Optional[pd.DataFrame]:
    """Pure-numpy fallback for Water Orientational Relaxation."""
    water = u.select_atoms(selection)
    if not water:
        return None
    
    # We'll compute P1(t) = <u(0) . u(t)> for O-H vectors
    # This is a simplified version for common water models (TIP3, TIP4, SPC)
    # It assumes H1 and H2 are the first two atoms after O in each residue.
    
    # Actually, let's do something more robust: find OH vectors for each residue
    res_oh_vectors = []
    for res in water.residues:
        o = res.atoms.select_atoms("name O*")
        h = res.atoms.select_atoms("name H*")
        if len(o) > 0 and len(h) > 0:
            # Take first O and first H for simplicity
            res_oh_vectors.append((o[0].index, h[0].index))
    
    if not res_oh_vectors:
        return None
    
    dt = 1.0 # default
    if len(u.trajectory) > 1:
        dt = u.trajectory.dt
    
    n_frames = len(range(options.frame_start, options.frame_stop, options.stride))
    if n_frames < 2:
        return None
        
    all_vectors = []
    times = []
    
    for ts in u.trajectory[options.frame_start:options.frame_stop:options.stride]:
        # Get all vectors for this frame
        pos = u.atoms.positions
        vecs = []
        for o_idx, h_idx in res_oh_vectors:
            v = pos[h_idx] - pos[o_idx]
            vecs.append(v / np.linalg.norm(v))
        all_vectors.append(np.array(vecs))
        times.append(ts.time)
        
    all_vectors = np.array(all_vectors) # (n_frames, n_waters, 3)
    
    # Compute autocorrelation
    # C(tau) = 1/(N * (T-tau)) sum_i sum_t u_i(t) . u_i(t+tau)
    # For speed, we'll just do a few tau points or use FFT
    n_res = all_vectors.shape[1]
    corrs = []
    taus = []
    
    # Use first few lag points for a quick plot
    max_lag = min(n_frames // 2, 50) 
    for lag in range(max_lag):
        if n_frames - lag <= 0: break
        # Dot product across all waters and frames
        v_t = all_vectors[:n_frames-lag]
        v_tau = all_vectors[lag:n_frames]
        # (v_t * v_tau).sum(axis=-1) -> (n_frames-lag, n_waters)
        c = (v_t * v_tau).sum(axis=-1).mean()
        corrs.append(c)
        taus.append(lag * dt * options.stride)
        
    return pd.DataFrame({"tau": taus, "correlation": corrs})

def compute_hbl_fallback_curve(u: mda.Universe, hbonds: np.ndarray, options, dt: float) -> Optional[pd.DataFrame]:
    """Pure-numpy fallback for H-bond lifetime correlation curve."""
    if not hbonds.size:
        return None
        
    # hbonds: [frame, d_idx, h_idx, a_idx, distance, angle]
    # We need to build a bit-map of which H-bonds exist in which frames
    unique_pairs = {} # (d, a) -> bitset or list of frames
    
    for row in hbonds:
        pair = (int(row[1]), int(row[3]))
        if pair not in unique_pairs:
            unique_pairs[pair] = set()
        unique_pairs[pair].add(int(row[0]))
        
    frames = sorted(list(set(hbonds[:, 0].astype(int))))
    if not frames:
        return None
        
    n_frames = len(frames)
    frame_map = {f: i for i, f in enumerate(frames)}
    
    # Bitmask for each pair: (unique_pairs, n_frames)
    mask = np.zeros((len(unique_pairs), n_frames), dtype=bool)
    for i, pair in enumerate(unique_pairs):
        for f in unique_pairs[pair]:
            if f in frame_map:
                mask[i, frame_map[f]] = True
                
    # Correlation C(tau) = <h(t)h(t+tau)> / <h(t)>
    corrs = []
    taus = []
    max_lag = min(n_frames // 2, 50)
    
    avg_h = mask.mean()
    if avg_h == 0:
        return None
        
    for lag in range(max_lag):
        if n_frames - lag <= 0: break
        m_t = mask[:, :n_frames-lag]
        m_tau = mask[:, lag:n_frames]
        # Intersection
        c = (m_t & m_tau).mean() / avg_h
        corrs.append(c)
        taus.append(lag * dt * options.stride)
        
    return pd.DataFrame({"tau": taus, "correlation": corrs})
