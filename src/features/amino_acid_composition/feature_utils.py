# feature_utils.py

import numpy as np
from AAlibraries import kD

# ---------------------------
#  Motif and hydrophobicity utilities
# ---------------------------

def find_nglyc_motifs(seq):
    """Find positions of N-glycosylation motifs (N-X-S/T, X ≠ P)."""
    positions = []
    for i in range(len(seq) - 2):
        if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in ["S", "T"]:
            positions.append(i + 1)
    return positions

def hydrophobicity_vector(seq):
    """Convert amino acid sequence into Kyte–Doolittle hydrophobicity values."""
    return [kD.get(aa, 0.0) for aa in seq]

# ---------------------------
#  Feature calculations
# ---------------------------

def compute_basic_features(seq):
    """Compute hydrophobicity and glycosylation motif features."""
    hydro = hydrophobicity_vector(seq)
    motifs = find_nglyc_motifs(seq)
    
    if len(seq) == 0:
        return {f: 0 for f in [
            "seq_len", "num_motifs", "motif_density",
            "avg_hydro", "std_hydro", "max_hydro", "min_hydro",
            "hydrophobic_fraction"
        ]}
    
    features = {
        "seq_len": len(seq),
        "num_motifs": len(motifs),
        "motif_density": len(motifs) / len(seq),
        "avg_hydro": np.mean(hydro),
        "std_hydro": np.std(hydro),
        "max_hydro": np.max(hydro),
        "min_hydro": np.min(hydro),
        "hydrophobic_fraction": np.sum(np.array(hydro) > 0) / len(hydro)
    }
    return features

def compute_aa_composition(seq):
    """Compute amino acid composition (fraction of each residue)."""
    aa_list = list(kD.keys())
    seq_len = len(seq)
    return {aa: seq.count(aa) / seq_len if seq_len > 0 else 0 for aa in aa_list}

def extract_all_features(seq):
    """Combine all feature groups into one dictionary."""
    feats = compute_basic_features(seq)
    feats.update(compute_aa_composition(seq))
    return feats
