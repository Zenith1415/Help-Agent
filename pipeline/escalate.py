"""
pipeline/escalate.py
Escalation Gate with Multi-Stage Cascade and Explicit Reason Logging.
Evaluates security/financial risk, customer sentiment, and model confidence.
"""

from typing import Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pipeline.intents import HARD_ESCALATE_INTENTS, SENTIMENT_SENSITIVE_INTENTS

CONFIDENCE_THRESHOLD = 0.60
SENTIMENT_NEGATIVE_THRESHOLD = -0.15


class EscalationGate:
    """
    Multi-stage escalation decision engine with reason logging.
    First match wins in the priority cascade.
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        sentiment_threshold: float = SENTIMENT_NEGATIVE_THRESHOLD
    ):
        self.confidence_threshold = confidence_threshold
        self.sentiment_threshold = sentiment_threshold
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Compute VADER sentiment scores for customer message."""
        scores = self.sentiment_analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound <= self.sentiment_threshold:
            label = "negative"
        elif compound >= 0.15:
            label = "positive"
        else:
            label = "neutral"
        return {"compound": round(compound, 3), "label": label}

    def evaluate(
        self,
        customer_msg: str,
        predicted_intent: str,
        confidence: float
    ) -> Dict[str, Any]:
        """
        Evaluate customer message through the 4-stage escalation cascade.
        Logs an auditable reason string for every decision.
        """
        sentiment_info = self.analyze_sentiment(customer_msg)
        sentiment_score = sentiment_info["compound"]
        sentiment_label = sentiment_info["label"]

        # Stage 1: Hard Rules for Security & Financial Risk
        if predicted_intent in HARD_ESCALATE_INTENTS:
            return {
                "should_escalate": True,
                "reason": f"escalated: intent={predicted_intent} (hard rule, security/financial risk)",
                "stage": 1,
                "sentiment": sentiment_info,
                "confidence": confidence
            }

        # Stage 2: Sentiment-Sensitive Escalation (frustration on sensitive intents)
        if sentiment_score <= self.sentiment_threshold and predicted_intent in SENTIMENT_SENSITIVE_INTENTS:
            return {
                "should_escalate": True,
                "reason": f"escalated: negative sentiment ({sentiment_score}) + intent={predicted_intent}",
                "stage": 2,
                "sentiment": sentiment_info,
                "confidence": confidence
            }

        # Stage 3: Low Classifier Confidence
        if confidence < self.confidence_threshold:
            return {
                "should_escalate": True,
                "reason": f"escalated: low confidence={confidence:.2f} < threshold={self.confidence_threshold:.2f}",
                "stage": 3,
                "sentiment": sentiment_info,
                "confidence": confidence
            }

        # Stage 4: Safe for Auto-Handling
        return {
            "should_escalate": False,
            "reason": f"auto: confidence={confidence:.2f}, sentiment={sentiment_label} ({sentiment_score})",
            "stage": 4,
            "sentiment": sentiment_info,
            "confidence": confidence
        }


_gate_instance = None

def get_escalation_gate() -> EscalationGate:
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = EscalationGate()
    return _gate_instance
