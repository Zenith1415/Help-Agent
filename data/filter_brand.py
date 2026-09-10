"""
data/filter_brand.py
Filter raw customer support data for @AmazonHelp, clean text, and reconstruct
(customer_msg, brand_reply) conversation pairs.
Supports both twcs.csv and conversations.parquet formats.
"""

import sys
import re
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

RAW_DIR = Path(__file__).resolve().parent / "raw"
RAW_CSV_PATH = RAW_DIR / "twcs.csv"
RAW_PARQUET_PATH = RAW_DIR / "conversations.parquet"

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"
OUTPUT_PAIRS_CSV = PROCESSED_DIR / "amazon_help_pairs.csv"

BRAND_AUTHOR_ID = "AmazonHelp"


def clean_tweet_text(text: str) -> str:
    """Normalize whitespace and newlines."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_handles(text: str) -> str:
    """Remove leading @mentions from customer tweet."""
    text = re.sub(r"^(@\w+\s*)+", "", text)
    return text.strip()


def is_likely_english(text: str) -> bool:
    """Fast heuristic to ensure tweets are English."""
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    common_en = {
        "the", "to", "and", "my", "is", "in", "for", "you", "it", "on",
        "with", "order", "have", "at", "this", "me", "not", "can", "amazon",
        "be", "was", "delivered", "package", "item", "refund", "help"
    }
    return len(words.intersection(common_en)) >= 2


def extract_turn_from_text(conv_text: str):
    """Extract first Customer inquiry and first Support reply."""
    c_match = re.search(r"Customer:\s*(.*?)(?=\nSupport:|\nCustomer:|$)", conv_text, re.DOTALL)
    s_match = re.search(r"Support:\s*(.*?)(?=\nCustomer:|\nSupport:|$)", conv_text, re.DOTALL)
    if c_match and s_match:
        c_text = clean_tweet_text(strip_handles(c_match.group(1)))
        s_text = clean_tweet_text(s_match.group(1))
        return c_text, s_text
    return None, None


def process_parquet(parquet_path: Path, output_path: Path, limit: int = None):
    """Extract English AmazonHelp pairs from conversations.parquet."""
    print(f"[i] Loading conversations from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    amz = df[df["company"] == BRAND_AUTHOR_ID]
    print(f"[OK] Found {len(amz)} {BRAND_AUTHOR_ID} conversations.")

    print("[i] Extracting first-turn conversation pairs and filtering for English...")
    pairs = []
    for conv_id, row in tqdm(amz.iterrows(), total=len(amz), desc="Extracting pairs"):
        conv = str(row["conversation"])
        c_msg, b_reply = extract_turn_from_text(conv)

        if not c_msg or not b_reply:
            continue
        if len(c_msg) < 15 or len(b_reply) < 15:
            continue
        if not is_likely_english(c_msg) or not is_likely_english(b_reply):
            continue

        pairs.append({
            "customer_msg": c_msg,
            "brand_reply": b_reply
        })

        if limit and len(pairs) >= limit:
            break

    df_pairs = pd.DataFrame(pairs)
    df_pairs = df_pairs.drop_duplicates(subset=["customer_msg"]).reset_index(drop=True)
    df_pairs.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Successfully saved {len(df_pairs)} cleaned English pairs to: {output_path}")


def process_csv(csv_path: Path, output_path: Path, limit: int = None, chunksize: int = 100_000):
    """Two-pass extraction on raw twcs.csv."""
    print(f"[1/3] Scanning {csv_path} for {BRAND_AUTHOR_ID} replies...")
    brand_replies = []
    needed_customer_ids = set()

    for chunk in pd.read_csv(
        csv_path,
        usecols=["tweet_id", "author_id", "inbound", "created_at", "text", "in_response_to_tweet_id"],
        chunksize=chunksize,
        low_memory=False,
        dtype={"tweet_id": str, "author_id": str, "inbound": bool, "text": str, "in_response_to_tweet_id": str}
    ):
        brand_chunk = chunk[
            (chunk["author_id"].str.lower() == BRAND_AUTHOR_ID.lower()) &
            (~chunk["inbound"]) &
            (chunk["in_response_to_tweet_id"].notna())
        ]
        for _, row in brand_chunk.iterrows():
            pid = str(row["in_response_to_tweet_id"]).strip().replace(".0", "")
            if pid and pid != "nan":
                needed_customer_ids.add(pid)
                brand_replies.append({
                    "brand_tweet_id": str(row["tweet_id"]),
                    "customer_tweet_id": pid,
                    "brand_reply": clean_tweet_text(row["text"])
                })
            if limit and len(brand_replies) >= limit:
                break
        if limit and len(brand_replies) >= limit:
            break

    print(f"[2/3] Matching {len(needed_customer_ids)} customer tweets...")
    customer_tweets = {}
    for chunk in pd.read_csv(
        csv_path,
        usecols=["tweet_id", "author_id", "inbound", "created_at", "text"],
        chunksize=chunksize,
        low_memory=False,
        dtype={"tweet_id": str, "author_id": str, "inbound": bool, "text": str}
    ):
        chunk["clean_id"] = chunk["tweet_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        matched = chunk[chunk["clean_id"].isin(needed_customer_ids)]
        for _, row in matched.iterrows():
            cid = row["clean_id"]
            if cid not in customer_tweets:
                customer_tweets[cid] = clean_tweet_text(strip_handles(str(row["text"])))
        if len(customer_tweets) >= len(needed_customer_ids):
            break

    print("[3/3] Pairing and filtering English messages...")
    paired_data = []
    for r in brand_replies:
        cid = r["customer_tweet_id"]
        if cid in customer_tweets:
            c_msg = customer_tweets[cid]
            b_reply = r["brand_reply"]
            if len(c_msg) < 15 or len(b_reply) < 15:
                continue
            if not is_likely_english(c_msg) or not is_likely_english(b_reply):
                continue
            paired_data.append({
                "customer_msg": c_msg,
                "brand_reply": b_reply
            })

    df_pairs = pd.DataFrame(paired_data)
    df_pairs = df_pairs.drop_duplicates(subset=["customer_msg"]).reset_index(drop=True)
    df_pairs.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Saved {len(df_pairs)} cleaned pairs to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Filter and extract AmazonHelp conversation pairs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum pairs to extract")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_PARQUET_PATH.exists():
        process_parquet(RAW_PARQUET_PATH, OUTPUT_PAIRS_CSV, limit=args.limit)
    elif RAW_CSV_PATH.exists():
        process_csv(RAW_CSV_PATH, OUTPUT_PAIRS_CSV, limit=args.limit)
    else:
        print("[!] No raw data found. Running data/download_data.py first...")
        from data.download_data import main as download_main
        download_main()
        if RAW_PARQUET_PATH.exists():
            process_parquet(RAW_PARQUET_PATH, OUTPUT_PAIRS_CSV, limit=args.limit)
        elif RAW_CSV_PATH.exists():
            process_csv(RAW_CSV_PATH, OUTPUT_PAIRS_CSV, limit=args.limit)


if __name__ == "__main__":
    main()
