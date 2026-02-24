"""
Data management for SecPipeAI.

CICIDS2017 and UNSW-NB15 cannot be downloaded automatically — they require
either manual download from the official institution websites or a Kaggle
account. This script:

  1. Checks which expected raw files are present in data/raw/<dataset>/
  2. Prints download instructions for any missing files
  3. Verifies SHA-256 checksums (from configs/checksums.yaml) for files
     that are present. A null checksum entry is treated as a soft warning,
     not a hard failure.
  4. Exits with code 1 if any required file is missing, so Makefile targets
     that depend on raw data will gate correctly.

Usage:
    python src/data/download.py --config configs/experiment.yaml
    make data
"""

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Expected file lists (exact filenames as distributed by the dataset authors)
# ---------------------------------------------------------------------------

CICIDS2017_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",                        # lowercase 'w'
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", # typo in official name
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

UNSW_NB15_FILES = [
    "UNSW_NB15_training-set.csv",
    "UNSW_NB15_testing-set.csv",
]

# ---------------------------------------------------------------------------
# Download instructions
# ---------------------------------------------------------------------------

CICIDS2017_INSTRUCTIONS = """\
  CICIDS2017 — download the "MachineLearningCVE" folder and place all CSVs in:
    data/raw/cicids2017/

  Option A — UNB CIC official:
    https://www.unb.ca/cic/datasets/ids-2017.html

  Option B — Kaggle:
    pip install kaggle
    kaggle datasets download -d cicdataset/cicids2017
    unzip cicids2017.zip -d data/raw/cicids2017/

  Option C — HuggingFace (community mirror, verify checksums after):
    pip install huggingface_hub
    huggingface-cli download --repo-type dataset \\
        "MrSamuelEklund/CICIDS2017" --local-dir data/raw/cicids2017/
"""

UNSW_NB15_INSTRUCTIONS = """\
  UNSW-NB15 — download training-set and testing-set CSVs and place them in:
    data/raw/unsw_nb15/

  Option A — UNSW official:
    https://research.unsw.edu.au/projects/unsw-nb15-dataset
    Download: UNSW_NB15_training-set.csv  UNSW_NB15_testing-set.csv

  Option B — Kaggle:
    pip install kaggle
    kaggle datasets download -d mrwellsdavid/unsw-nb15
    unzip unsw-nb15.zip -d data/raw/unsw_nb15/

  Option C — HuggingFace (community mirror, verify checksums after):
    pip install huggingface_hub
    huggingface-cli download --repo-type dataset \\
        "dhruv1234/UNSW-NB15" --local-dir data/raw/unsw_nb15/
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: Path, expected: str | None) -> bool:
    """Returns True on pass or soft-pass (expected is None). Raises ValueError on mismatch."""
    if expected is None:
        print(f"  WARNING: no checksum registered for {path.name} — skipping verification.")
        print(f"    To register: sha256sum {path}")
        return True
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(
            f"Checksum MISMATCH for {path.name}:\n"
            f"  expected : {expected}\n"
            f"  computed : {actual}\n"
            f"  The file may be corrupt or from a different source."
        )
    print(f"  OK  {path.name}")
    return True


def check_dataset(
    slug: str,
    raw_dir: Path,
    expected_files: list[str],
    checksums: dict,
) -> tuple[list[str], list[str]]:
    """Returns (present, missing) filename lists. Verifies checksums for present files."""
    present, missing = [], []
    dataset_checksums = checksums.get(slug, {}) or {}

    for fname in expected_files:
        fpath = raw_dir / fname
        if fpath.exists():
            present.append(fname)
            expected_hash = dataset_checksums.get(fname)
            verify_checksum(fpath, expected_hash)
        else:
            missing.append(fname)

    return present, missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    raw_root = Path(cfg["paths"]["raw_data"])
    checksums_path = Path(cfg["paths"]["checksums"])

    checksums: dict = {}
    if checksums_path.exists():
        with checksums_path.open() as fh:
            checksums = yaml.safe_load(fh) or {}
    else:
        print(f"WARNING: checksums file not found at {checksums_path}; skipping verification.")

    all_ok = True

    # ---- CICIDS2017 ----
    cic_dir = raw_root / cfg["datasets"]["cicids2017"]["raw_subdir"]
    print(f"\n[CICIDS2017] Checking {cic_dir} ...")
    cic_dir.mkdir(parents=True, exist_ok=True)
    present, missing = check_dataset("cicids2017", cic_dir, CICIDS2017_FILES, checksums)
    print(f"  {len(present)}/{len(CICIDS2017_FILES)} files present.")
    if missing:
        all_ok = False
        print(f"\n  MISSING {len(missing)} file(s):")
        for f in missing:
            print(f"    - {f}")
        print(f"\n{CICIDS2017_INSTRUCTIONS}")

    # ---- UNSW-NB15 ----
    unsw_dir = raw_root / cfg["datasets"]["unsw_nb15"]["raw_subdir"]
    print(f"\n[UNSW-NB15] Checking {unsw_dir} ...")
    unsw_dir.mkdir(parents=True, exist_ok=True)
    present, missing = check_dataset("unsw_nb15", unsw_dir, UNSW_NB15_FILES, checksums)
    print(f"  {len(present)}/{len(UNSW_NB15_FILES)} files present.")
    if missing:
        all_ok = False
        print(f"\n  MISSING {len(missing)} file(s):")
        for f in missing:
            print(f"    - {f}")
        print(f"\n{UNSW_NB15_INSTRUCTIONS}")

    print()
    if all_ok:
        print("All raw data present and checksums verified.")
        print("REMINDER: Do NOT modify any file under data/raw/")
    else:
        print("One or more dataset files are missing. See instructions above.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check raw dataset presence and verify checksums"
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    main(args.config)
