"""
eval/build_golden_set.py
Carves off an untouchable holdout pool from the raw dataset, then constructs
a stratified, high-quality golden evaluation set (160-200 examples across 8 intents)
with ground truth intent, escalation decision, and rationale.
"""

import sys
import json
import re
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ALL_PAIRS_CSV = PROCESSED_DIR / "amazon_help_pairs.csv"
TRAIN_POOL_CSV = PROCESSED_DIR / "train_pool.csv"
HOLDOUT_POOL_CSV = PROCESSED_DIR / "holdout_pool.csv"

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_SET_JSONL = EVAL_DIR / "golden_set.jsonl"

RANDOM_SEED = 42

# Intent detection keywords for stratified candidate sampling
INTENT_KEYWORDS = {
    "account_access": [
        "locked", "hack", "login", "log in", "password", "blocked my account",
        "access to my account", "otp", "deactivated account", "sign in"
    ],
    "billing_payment": [
        "charged twice", "unauthorized charge", "double charge", "gift card balance",
        "withheld", "bank account", "deducted", "payment failed", "credit card", "cashback"
    ],
    "damaged_defective": [
        "broken", "damaged", "shattered", "tampered", "wrong item", "missing item",
        "scratched", "defective", "opened box", "incorrect product"
    ],
    "order_delivery": [
        "track", "where is my order", "not delivered", "late delivery", "2 day delivery",
        "dispatch", "courier", "driver", "delayed", "haven't received", "stolen"
    ],
    "refund_return": [
        "refund", "return", "return label", "money back", "cancelled but not refunded",
        "pickup return", "send back", "return item"
    ],
    "subscription_prime": [
        "prime membership", "paying for prime", "cancel prime", "prime video",
        "prime delivery", "student prime", "annual prime", "renew"
    ],
    "technical_product": [
        "alexa", "echo", "fire tv", "firestick", "kindle edition", "app crash",
        "smart home", "device", "ebook", "kindle app"
    ],
    "complaint_feedback": [
        "worst service", "pathetic", "disgusted", "terrible", "shame on you",
        "useless", "incompetent", "thank you", "great service", "you rock"
    ]
}


def split_train_holdout(all_pairs_path: Path, holdout_size: int = 10000):
    """Split processed pairs into train pool and untouched holdout pool."""
    print(f"[i] Loading processed pairs from {all_pairs_path}...")
    df = pd.read_csv(all_pairs_path)
    df = df.dropna(subset=["customer_msg", "brand_reply"]).drop_duplicates(subset=["customer_msg"])

    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_size,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    train_df.to_csv(TRAIN_POOL_CSV, index=False, encoding="utf-8")
    holdout_df.to_csv(HOLDOUT_POOL_CSV, index=False, encoding="utf-8")

    print(f"[OK] Split dataset into:")
    print(f"     - Train pool  : {len(train_df):,} pairs -> {TRAIN_POOL_CSV.name}")
    print(f"     - Holdout pool: {len(holdout_df):,} pairs -> {HOLDOUT_POOL_CSV.name}")
    return holdout_df


def identify_candidate_intent(text: str) -> str:
    """Classify message intent based on keyword priority."""
    text_lower = text.lower()
    
    # Check security & billing first
    for intent in ["account_access", "billing_payment", "damaged_defective", "refund_return", "subscription_prime", "technical_product", "order_delivery", "complaint_feedback"]:
        keywords = INTENT_KEYWORDS[intent]
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                return intent
    return "order_delivery"


def decide_escalation(intent: str, text: str):
    """Determine ground truth escalation decision and rationale."""
    text_lower = text.lower()

    if intent == "account_access":
        return True, "escalate: account security/login failure requires human verification"
    if intent == "billing_payment":
        return True, "escalate: financial dispute / unauthorized charge requires billing team"
    if intent == "damaged_defective":
        return True, "escalate: damaged or tampered physical goods require human exception handling"

    # Extreme anger / legal keywords
    angry_keywords = ["scam", "fraud", "lawsuit", "consumer court", "police", "stolen", "pathetic", "disgusted", "incompetent", "lied"]
    if any(ak in text_lower for ak in angry_keywords):
        return True, f"escalate: high hostility / negative sentiment for {intent}"

    if intent == "refund_return" and ("not refunded" in text_lower or "money back" in text_lower or "month" in text_lower):
        return True, "escalate: disputed / delayed refund requires human investigation"

    if intent in ["order_delivery", "subscription_prime", "technical_product"]:
        return False, f"auto: standard informational inquiry for {intent}"

    return False, f"auto: standard resolution for {intent}"


def build_golden_set(holdout_df: pd.DataFrame, target_per_intent: int = 22):
    """Construct stratified 160-200 example golden evaluation set."""
    print(f"\n[i] Curating stratified golden evaluation set (~{target_per_intent} per intent)...")

    # Group holdout examples by candidate intent
    intent_candidates = {intent: [] for intent in INTENT_KEYWORDS.keys()}

    for _, row in holdout_df.iterrows():
        c_msg = str(row["customer_msg"]).strip()
        b_reply = str(row["brand_reply"]).strip()
        
        # Quality filters
        if len(c_msg) < 25 or len(c_msg) > 300:
            continue
        if "http" in c_msg and len(c_msg) < 50:
            continue

        pred_intent = identify_candidate_intent(c_msg)
        intent_candidates[pred_intent].append((c_msg, b_reply))

    golden_examples = []
    ex_id = 1

    for intent, items in intent_candidates.items():
        selected = items[:target_per_intent]
        print(f"     - {intent:20s}: {len(selected)} examples selected")
        for c_msg, b_reply in selected:
            should_esc, esc_reason = decide_escalation(intent, c_msg)
            is_ambiguous = any(kw in c_msg.lower() for kw in ["delayed", "stolen", "refund", "charge", "broken"]) and intent not in ["damaged_defective", "billing_payment"]
            golden_examples.append({
                "id": f"gold_{ex_id:03d}",
                "customer_msg": c_msg,
                "historical_reply": b_reply,
                "true_intent": intent,
                "should_escalate": should_esc,
                "escalation_reason": esc_reason,
                "is_ambiguous": bool(is_ambiguous)
            })
            ex_id += 1

    # Save to jsonl
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_SET_JSONL, "w", encoding="utf-8") as f:
        for ex in golden_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n[OK] Golden evaluation set created with {len(golden_examples)} examples:")
    print(f"     File: {GOLDEN_SET_JSONL}")
    
    # Summary of escalation distribution
    esc_count = sum(1 for e in golden_examples if e["should_escalate"])
    print(f"     Escalations: {esc_count}/{len(golden_examples)} ({(esc_count/len(golden_examples))*100:.1f}%)")
    print(f"     Auto-handled: {len(golden_examples) - esc_count}/{len(golden_examples)} ({((len(golden_examples)-esc_count)/len(golden_examples))*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Build train/holdout split and golden evaluation set.")
    parser.add_argument("--force-split", action="store_true", help="Force re-split of dataset")
    args = parser.parse_args()

    if not HOLDOUT_POOL_CSV.exists() or args.force_split:
        holdout_df = split_train_holdout(ALL_PAIRS_CSV)
    else:
        print(f"[OK] Found existing holdout pool at {HOLDOUT_POOL_CSV.name}")
        holdout_df = pd.read_csv(HOLDOUT_POOL_CSV)

    build_golden_set(holdout_df, target_per_intent=22)


if __name__ == "__main__":
    main()
