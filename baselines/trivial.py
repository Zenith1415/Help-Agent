"""
baselines/trivial.py
Trivial Baseline:
- Intent: Always predicts the majority class ("order_delivery").
- Reply: Static canned reply for that intent.
- Escalation: Always escalates to a human agent.
"""

from typing import Dict, Any
from pipeline.intents import INTENTS

MAJORITY_INTENT = "order_delivery"

CANNED_REPLIES = {
    "order_delivery": "Thanks for contacting Amazon Help! Please check your order status on Amazon.com under 'Your Orders' or track your package with the carrier link provided in your shipping confirmation email.",
    "refund_return": "Thanks for reaching out! To start a return or check your refund status, please visit the Online Return Center on Amazon.",
    "damaged_defective": "We're sorry your item arrived in less than perfect condition! Please visit 'Your Orders' and select 'Return or replace items' to request a free replacement.",
    "account_access": "For your security, please reset your password on the sign-in page or contact our customer verification team directly.",
    "billing_payment": "For billing disputes or unauthorized charges, please review your Payment Options in your account settings or contact our billing specialists.",
    "subscription_prime": "You can manage your Prime membership settings and view benefits anytime at amazon.com/prime.",
    "technical_product": "Please try restarting your device and ensuring your app is updated to the latest software version.",
    "complaint_feedback": "Thank you for your feedback. We appreciate your patience and will share this with our leadership team."
}


class TrivialBaseline:
    """Trivial heuristic baseline."""

    def __init__(self):
        self.majority_intent = MAJORITY_INTENT

    def classify_intent(self, text: str) -> Dict[str, Any]:
        """Always returns the majority class with fixed confidence."""
        return {
            "intent": self.majority_intent,
            "confidence": 0.50,
            "model": "trivial_majority_class"
        }

    def draft_reply(self, text: str, intent: str = MAJORITY_INTENT) -> str:
        """Returns canned static reply for the given intent."""
        return CANNED_REPLIES.get(intent, CANNED_REPLIES[MAJORITY_INTENT])

    def decide_escalation(self, text: str, intent: str = MAJORITY_INTENT) -> Dict[str, Any]:
        """Trivial policy: Always escalate to human agent."""
        return {
            "should_escalate": True,
            "reason": "trivial: always escalate baseline",
            "model": "trivial_always_escalate"
        }
