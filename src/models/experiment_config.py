from pathlib import Path
import os


SPECIES = os.getenv("SPECIES", "Diabates")
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed" / SPECIES
REPORTS_DIR = ROOT / "reports" / "results" / SPECIES
FEATURE_ROOT = ROOT / "artifacts" / "features"

FEATURE_COMPONENTS = {
    "aac": ("aac",),
    "ct": ("ct",),
    "pssm": ("pssm",),
    "aac_ct": ("aac", "ct"),
    "aac_pssm": ("aac", "pssm"),
    "ct_pssm": ("ct", "pssm"),
    "aac_ct_pssm": ("aac", "ct", "pssm"),
}

COMPONENT_DIRS = {
    "aac": FEATURE_ROOT / "amino_acid_composition" / SPECIES / "Profiles",
    "ct": FEATURE_ROOT / "conjoint_triads" / SPECIES / "Profiles",
    "pssm": FEATURE_ROOT / "pssm" / SPECIES / "Profiles",
}

DEFAULT_FEATURE_SETS = ["aac", "ct", "pssm", "aac_ct", "aac_pssm", "ct_pssm", "aac_ct_pssm"]
DEFAULT_SPLITS = ["random_pair", "protein_holdout", "low_similarity_holdout"]
DEFAULT_MODELS = [
    "similarity",
    "degree",
    "logistic",
    "random_forest",
    "xgboost",
    "lightgbm",
    "rotation_forest",
]
