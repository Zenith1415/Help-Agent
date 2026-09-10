"""
pipeline/retrieve.py
In-memory dense vector retrieval using FAISS and sentence-transformers/all-MiniLM-L6-v2.
Searches over historical (customer_msg, brand_reply) pairs from the training pool.
"""

import sys
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TRAIN_POOL_CSV = INDEX_DIR / "train_pool.csv"
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
METADATA_FILE = INDEX_DIR / "faiss_metadata.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_INDEX_SIZE = 15000  # Index 15k high-quality pairs for fast retrieval and low memory


class Retriever:
    """FAISS-based historical support pair retriever."""

    def __init__(self, index_size: int = DEFAULT_INDEX_SIZE):
        self.index_size = index_size
        self.model = SentenceTransformer(MODEL_NAME)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, str]] = []
        self._load_or_build_index()

    def _load_or_build_index(self):
        """Loads cached FAISS index or builds from train_pool.csv."""
        if INDEX_FILE.exists() and METADATA_FILE.exists():
            print(f"[i] Loading FAISS index from {INDEX_FILE.name}...")
            self.index = faiss.read_index(str(INDEX_FILE))
            with open(METADATA_FILE, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[OK] Loaded FAISS index with {self.index.ntotal} historical pairs.")
            return

        SEED_CSV = INDEX_DIR / "retrieval_seed_pairs.csv"
        if TRAIN_POOL_CSV.exists():
            print(f"[i] Building FAISS index from {TRAIN_POOL_CSV.name}...")
            df = pd.read_csv(TRAIN_POOL_CSV).dropna(subset=["customer_msg", "brand_reply"])
        elif SEED_CSV.exists():
            print(f"[i] Building FAISS index from {SEED_CSV.name} (seed repository fallback)...")
            df = pd.read_csv(SEED_CSV).dropna(subset=["customer_msg", "brand_reply"])
        else:
            raise FileNotFoundError(f"Neither {TRAIN_POOL_CSV.name} nor {SEED_CSV.name} found. Run data/download_data.py first.")

        df = df.drop_duplicates(subset=["customer_msg"]).head(self.index_size)

        # Exclude generic DM redirect replies if possible to favor grounded replies
        df["is_boilerplate"] = df["brand_reply"].str.lower().str.contains(r"please send us a dm|dm us your details|click here to dm", regex=True)
        # Sort so informative replies are prioritized
        df = df.sort_values(by="is_boilerplate", ascending=True)

        customer_texts = df["customer_msg"].tolist()
        brand_replies = df["brand_reply"].tolist()

        print(f"[i] Encoding {len(customer_texts)} customer inquiries with {MODEL_NAME}...")
        embeddings = self.model.encode(
            customer_texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dim = embeddings.shape[1]
        # Inner product on normalized embeddings = cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

        self.metadata = [
            {"customer_msg": c, "brand_reply": b}
            for c, b in zip(customer_texts, brand_replies)
        ]

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_FILE))
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[OK] Built and saved FAISS index with {self.index.ntotal} pairs.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top_k most similar historical customer inquiries and their brand replies."""
        query_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self.index.search(query_emb.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                item = dict(self.metadata[idx])
                item["similarity"] = float(score)
                results.append(item)
        return results


# Global singleton instance
_retriever_instance = None

def get_retriever() -> Retriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance
