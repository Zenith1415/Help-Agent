"""
data/download_data.py
Fetch customer support conversations for AmazonHelp.
Supports both Kaggle API (thoughtvector/customer-support-on-twitter) and
direct HuggingFace mirror fallback (TNE-AI/customer-support-on-twitter-conversation).
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

KAGGLE_DATASET_NAME = "thoughtvector/customer-support-on-twitter"
HF_MIRROR_URL = "https://huggingface.co/datasets/TNE-AI/customer-support-on-twitter-conversation/resolve/main/data/train-00000-of-00001.parquet"

RAW_DATA_DIR = Path(__file__).resolve().parent / "raw"
TARGET_CSV = RAW_DATA_DIR / "twcs.csv"
TARGET_PARQUET = RAW_DATA_DIR / "conversations.parquet"


def check_existing_data() -> bool:
    """Check if twcs.csv or conversations.parquet is already present."""
    if TARGET_CSV.exists() and TARGET_CSV.stat().st_size > 0:
        print(f"[OK] Found existing dataset at: {TARGET_CSV} ({TARGET_CSV.stat().st_size / (1024*1024):.1f} MB)")
        return True
    if TARGET_PARQUET.exists() and TARGET_PARQUET.stat().st_size > 0:
        print(f"[OK] Found existing dataset at: {TARGET_PARQUET} ({TARGET_PARQUET.stat().st_size / (1024*1024):.1f} MB)")
        return True
    return False


def download_via_kaggle() -> bool:
    """Attempt download using Kaggle API."""
    print(f"[i] Checking Kaggle API configuration for '{KAGGLE_DATASET_NAME}'...")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env = os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")

    if not kaggle_json.exists() and not has_env:
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[i] Downloading from Kaggle into {RAW_DATA_DIR}...")
        api.dataset_download_files(KAGGLE_DATASET_NAME, path=str(RAW_DATA_DIR), unzip=True)
        return TARGET_CSV.exists()
    except Exception as e:
        print(f"[!] Kaggle download error: {e}")
        return False


def download_via_hf_mirror() -> bool:
    """Download directly from the open HuggingFace mirror."""
    import urllib.request
    from tqdm import tqdm

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[i] Downloading dataset from open mirror:\n    {HF_MIRROR_URL}")

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    try:
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(HF_MIRROR_URL, filename=TARGET_PARQUET, reporthook=t.update_to)
        print(f"[OK] Saved dataset to {TARGET_PARQUET} ({TARGET_PARQUET.stat().st_size / (1024*1024):.1f} MB)")
        return True
    except Exception as e:
        print(f"[X] Failed downloading mirror: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download customer support dataset.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist.")
    parser.add_argument("--source", choices=["auto", "kaggle", "mirror"], default="auto", help="Download source")
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.force and check_existing_data():
        return

    success = False
    if args.source == "kaggle" or (args.source == "auto" and (os.getenv("KAGGLE_USERNAME") or (Path.home() / ".kaggle" / "kaggle.json").exists())):
        success = download_via_kaggle()

    if not success and args.source in ["auto", "mirror"]:
        print("[i] Falling back to open dataset mirror (no credentials required)...")
        success = download_via_hf_mirror()

    if not success:
        print("[X] Could not acquire dataset. Please check your network connection or provide Kaggle credentials.")
        sys.exit(1)


if __name__ == "__main__":
    main()
