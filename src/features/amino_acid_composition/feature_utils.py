# feature_utils.py

import numpy as np

from AAlibraries import kD


def find_nglyc_motifs(seq: str) -> list[int]:
    positions = []
    for index in range(len(seq) - 2):
        if seq[index] == "N" and seq[index + 1] != "P" and seq[index + 2] in ["S", "T"]:
            positions.append(index + 1)
    return positions


def hydrophobicity_vector(seq: str) -> list[float]:
    return [kD.get(aa, 0.0) for aa in seq]


def compute_basic_features(seq: str) -> dict[str, float]:
    hydro = hydrophobicity_vector(seq)
    motifs = find_nglyc_motifs(seq)

    if len(seq) == 0:
        return {
            "seq_len": 0,
            "num_motifs": 0,
            "motif_density": 0,
            "avg_hydro": 0,
            "std_hydro": 0,
            "max_hydro": 0,
            "min_hydro": 0,
            "hydrophobic_fraction": 0,
        }

    return {
        "seq_len": len(seq),
        "num_motifs": len(motifs),
        "motif_density": len(motifs) / len(seq),
        "avg_hydro": float(np.mean(hydro)),
        "std_hydro": float(np.std(hydro)),
        "max_hydro": float(np.max(hydro)),
        "min_hydro": float(np.min(hydro)),
        "hydrophobic_fraction": float(np.sum(np.array(hydro) > 0) / len(hydro)),
    }


def compute_aa_composition(seq: str) -> dict[str, float]:
    aa_list = list(kD.keys())
    seq_len = len(seq)
    return {aa: seq.count(aa) / seq_len if seq_len > 0 else 0 for aa in aa_list}
