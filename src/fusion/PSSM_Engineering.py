import os
from pathlib import Path

SPECIES = os.getenv("SPECIES", "Diabates")
ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    print(
        "PSSM_Engineering.py is no longer needed. "
        "Fusion now computes the mean PSSM vector directly from "
        f"{ROOT / 'artifacts' / 'features' / 'pssm' / SPECIES / 'Profiles'}."
    )


if __name__ == "__main__":
    main()
