"""
baselines/simple.py
Simple Baseline:
- Intent: TF-IDF Vectorizer + Logistic Regression.
- Reply: Per-intent template populated with nearest historical response phrase.
- Escalation: Confidence threshold only (escalate if confidence < 0.65).
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pipeline.intents import INTENTS

MODEL_DIR = Path(__file__).resolve().parent / "models"
TFIDF_MODEL_PATH = MODEL_DIR / "simple_tfidf_lr.pkl"
TRAIN_POOL_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "train_pool.csv"

# Per-intent response templates
TEMPLATES = {
    "order_delivery": "Hello! I understand you are inquiring about your delivery. According to our standard policy, you can track your package updates and delivery ETA directly in 'Your Orders' online.",
    "refund_return": "Hello! Regarding your refund/return request, refunds typically take 3-5 business days once the item reaches our fulfillment center.",
    "damaged_defective": "I am very sorry to hear that your item was received damaged or defective. You can initiate a free replacement or return right from your order details page.",
    "account_access": "Hello. For security reasons regarding your account login, please use our 2-step verification recovery page or reset your password.",
    "billing_payment": "Hello. For any unauthorized or unexpected charges, please check your digital orders and payment statement to review the transaction details.",
    "subscription_prime": "Hello! You can manage your Prime membership benefits, payment settings, and renewals directly at amazon.com/prime.",
    "technical_product": "Hello! For assistance with your Amazon device or digital content, please ensure your device is connected to Wi-Fi and updated to the latest software version.",
    "complaint_feedback": "Hello. We sincerely apologize for your frustrating experience. We are committed to providing the highest quality customer care."
}


class SimpleBaseline:
    """TF-IDF + Logistic Regression Classifier and Template Reply System."""

    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None
        self._load_or_train()

    def _load_or_train(self):
        """Loads cached model or trains a fast model on labeled training pairs."""
        if TFIDF_MODEL_PATH.exists():
            with open(TFIDF_MODEL_PATH, "rb") as f:
                data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.classifier = data["classifier"]
            return

        print("[i] Training simple TF-IDF + Logistic Regression baseline...")
        from eval.build_golden_set import identify_candidate_intent

        SEED_POOL_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "retrieval_seed_pairs.csv"
        if TRAIN_POOL_PATH.exists():
            df_train = pd.read_csv(TRAIN_POOL_PATH).dropna(subset=["customer_msg"])
        elif SEED_POOL_PATH.exists():
            df_train = pd.read_csv(SEED_POOL_PATH).dropna(subset=["customer_msg"])
        else:
            raise FileNotFoundError(f"Neither {TRAIN_POOL_PATH.name} nor {SEED_POOL_PATH.name} found.")

        # Sample training examples for quick, balanced training
        df_sample = df_train.sample(n=min(4000, len(df_train)), random_state=42).copy()
        
        # Weak-supervision labels using taxonomy rules
        df_sample["label"] = df_sample["customer_msg"].apply(identify_candidate_intent)

        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(df_sample["customer_msg"])
        y = df_sample["label"]

        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        self.classifier.fit(X, y)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(TFIDF_MODEL_PATH, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "classifier": self.classifier}, f)
        print(f"[OK] Trained and saved simple baseline to {TFIDF_MODEL_PATH.name}")

    def classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify message intent using TF-IDF + Logistic Regression."""
        X_vec = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(X_vec)[0]
        best_idx = np.argmax(probs)
        intent = self.classifier.classes_[best_idx]
        confidence = float(probs[best_idx])

        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "model": "simple_tfidf_lr"
        }

    def draft_reply(self, text: str, intent: str) -> str:
        """Returns template reply for the predicted intent."""
        return TEMPLATES.get(intent, TEMPLATES["order_delivery"])

    def decide_escalation(self, text: str, intent: str, confidence: float) -> Dict[str, Any]:
        """Escalate strictly if classifier confidence falls below threshold."""
        if confidence < self.confidence_threshold:
            return {
                "should_escalate": True,
                "reason": f"simple: low confidence threshold ({confidence:.2f} < {self.confidence_threshold:.2f})",
                "model": "simple_threshold_escalate"
            }
        return {
            "should_escalate": False,
            "reason": f"simple: confidence adequate ({confidence:.2f} >= {self.confidence_threshold:.2f})",
            "model": "simple_threshold_escalate"
        }
