"""
data/cluster_intents.py
Embed customer messages with sentence-transformers and cluster with KMeans (k=10)
to derive the empirical intent taxonomy from AmazonHelp customer inquiries.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

PAIRS_CSV = Path(__file__).resolve().parent / "processed" / "amazon_help_pairs.csv"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"
SUMMARY_CSV = PROCESSED_DIR / "cluster_summary.csv"
EXAMPLES_CSV = PROCESSED_DIR / "cluster_examples.csv"


def extract_top_tfidf_words(texts_by_cluster, n_top_words=10):
    """Extract top distinct words for each cluster using TF-IDF."""
    cluster_docs = [" ".join(texts) for texts in texts_by_cluster]
    
    vectorizer = TfidfVectorizer(
        max_df=0.85,
        min_df=2,
        stop_words="english",
        ngram_range=(1, 2)
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(cluster_docs)
        feature_names = np.array(vectorizer.get_feature_names_out())
        
        top_terms = []
        for i in range(len(cluster_docs)):
            row = tfidf_matrix[i].toarray().flatten()
            if row.sum() == 0:
                top_terms.append("N/A")
            else:
                top_indices = row.argsort()[::-1][:n_top_words]
                top_terms.append(", ".join(feature_names[top_indices]))
        return top_terms
    except Exception as e:
        return [f"Error extracting keywords: {e}"] * len(cluster_docs)


def cluster_messages(
    input_csv: Path,
    k: int = 10,
    sample_size: int = 5000,
    random_state: int = 42
):
    """Run KMeans clustering on customer message embeddings and export summaries."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}. Run data/filter_brand.py first.")

    print(f"[1/4] Loading customer messages from {input_csv}...")
    df = pd.read_csv(input_csv)
    if "customer_msg" not in df.columns:
        raise ValueError("CSV must contain a 'customer_msg' column.")

    df = df.dropna(subset=["customer_msg"]).drop_duplicates(subset=["customer_msg"])
    if len(df) > sample_size:
        print(f"[i] Subsampling {sample_size} customer messages for clustering...")
        df_sample = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    else:
        df_sample = df.reset_index(drop=True)

    print(f"[2/4] Embedding {len(df_sample)} messages using 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        df_sample["customer_msg"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"[3/4] Fitting KMeans (k={k})...")
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(embeddings)
    df_sample["cluster"] = cluster_labels

    distances = np.linalg.norm(embeddings - kmeans.cluster_centers_[cluster_labels], axis=1)
    df_sample["dist_to_center"] = distances

    print("[4/4] Generating cluster summary & representative examples...")
    cluster_texts = []
    summary_rows = []
    
    for c_id in range(k):
        c_subset = df_sample[df_sample["cluster"] == c_id]
        cluster_texts.append(c_subset["customer_msg"].tolist())
        summary_rows.append({
            "cluster_id": c_id,
            "count": len(c_subset),
            "percentage": f"{(len(c_subset) / len(df_sample)) * 100:.1f}%"
        })

    top_keywords = extract_top_tfidf_words(cluster_texts, n_top_words=8)
    for i, row in enumerate(summary_rows):
        row["top_keywords"] = top_keywords[i]

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[OK] Saved cluster summary to: {SUMMARY_CSV}")

    example_rows = []
    for c_id in range(k):
        c_subset = df_sample[df_sample["cluster"] == c_id].sort_values("dist_to_center").head(8)
        for _, row in c_subset.iterrows():
            example_rows.append({
                "cluster_id": c_id,
                "dist_to_center": f"{row['dist_to_center']:.3f}",
                "customer_msg": row["customer_msg"]
            })

    df_examples = pd.DataFrame(example_rows)
    df_examples.to_csv(EXAMPLES_CSV, index=False)
    print(f"[OK] Saved representative cluster examples to: {EXAMPLES_CSV}")
    print("\n--- Cluster Keyword Summary ---")
    print(df_summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Cluster customer messages to derive intent taxonomy.")
    parser.add_argument("--input", type=str, default=str(PAIRS_CSV), help="Path to amazon_help_pairs.csv")
    parser.add_argument("--k", type=int, default=10, help="Number of KMeans clusters (default: 10)")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of customer messages to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cluster_messages(
        input_csv=Path(args.input),
        k=args.k,
        sample_size=args.sample_size,
        random_state=args.seed
    )


if __name__ == "__main__":
    main()
