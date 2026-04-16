import os
from pathlib import Path

import pandas as pd


SPECIES = os.getenv("SPECIES", "Diabates")
ROOT = Path(__file__).resolve().parents[2]
AAC_DIR = ROOT / "artifacts" / "features" / "amino_acid_composition" / SPECIES / "Profiles"
CT_DIR = ROOT / "artifacts" / "features" / "conjoint_triads" / SPECIES / "Profiles"
PSSM_DIR = ROOT / "artifacts" / "features" / "pssm" / SPECIES / "Profiles"
FUSION_NAME = os.getenv("FUSION_NAME", "aac_ct_pssm").lower()
FUSION_COMPONENTS = {
    "aac_ct": ("aac", "ct"),
    "aac_pssm": ("aac", "pssm"),
    "ct_pssm": ("ct", "pssm"),
    "aac_ct_pssm": ("aac", "ct", "pssm"),
}
OUTPUT_DIR = ROOT / "artifacts" / "fusion" / SPECIES / FUSION_NAME / "Profiles"
MASTER_CSV = ROOT / "artifacts" / "fusion" / SPECIES / FUSION_NAME / "feature_vectors.csv"


def get_parquet_files(folder: Path) -> dict[str, Path]:
    return {path.name: path for path in folder.glob("*.parquet")}


def load_component(component: str, file_name: str) -> pd.DataFrame:
    if component == "aac":
        return pd.read_parquet((AAC_DIR / file_name)).copy().add_prefix("AAC_")
    if component == "ct":
        return pd.read_parquet((CT_DIR / file_name)).copy()
    if component == "pssm":
        pssm_df = pd.read_parquet(PSSM_DIR / file_name).copy()
        return pd.DataFrame([pssm_df.mean(axis=0)]).add_prefix("PSSM_")
    raise ValueError(f"Unknown component: {component}")


def main() -> None:
    components = FUSION_COMPONENTS.get(FUSION_NAME)
    if components is None:
        raise ValueError(f"Unsupported FUSION_NAME '{FUSION_NAME}'. Expected one of: {', '.join(FUSION_COMPONENTS)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = {
        "aac": get_parquet_files(AAC_DIR),
        "ct": get_parquet_files(CT_DIR),
        "pssm": get_parquet_files(PSSM_DIR),
    }
    common_files = set.intersection(*(set(sources[component]) for component in components))

    print(f"Building fusion '{FUSION_NAME}' with {len(common_files)} common proteins.")
    fused_rows = []

    for file_name in sorted(common_files):
        try:
            feature_frames = [load_component(component, file_name) for component in components]
            fused_df = pd.concat(feature_frames, axis=1)
            fused_df.to_parquet(OUTPUT_DIR / file_name, index=False)

            pid = Path(file_name).stem
            fused_df.insert(0, "Protein Identifier", pid)
            fused_rows.append(fused_df)
        except Exception as exc:
            print(f"Error processing {file_name}: {exc}")

    if not fused_rows:
        print(f"No matching proteins found for fusion '{FUSION_NAME}'.")
        return

    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_master = pd.concat(fused_rows, ignore_index=True)
    df_master.to_csv(MASTER_CSV, index=False)
    print(f"Fusion complete for '{FUSION_NAME}'. Saved {len(fused_rows)} fused profiles to: {MASTER_CSV}")


if __name__ == "__main__":
    main()
