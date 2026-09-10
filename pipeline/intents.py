"""
pipeline/intents.py
Single Source of Truth for Intent Taxonomy, Definitions, Examples, and Escalation Rules.
Derived empirically from clustering 53,511 AmazonHelp customer inquiries.
"""

from typing import Dict, List, Any

# Canonical List of 8 Intents
INTENTS = [
    "order_delivery",
    "refund_return",
    "damaged_defective",
    "account_access",
    "billing_payment",
    "subscription_prime",
    "technical_product",
    "complaint_feedback"
]

# Hard Escalation Intents: Always escalate to a human agent (security & financial risk)
HARD_ESCALATE_INTENTS = {
    "account_access",
    "billing_payment"
}

# Sentiment-Sensitive Escalation Intents: Escalate if customer sentiment is negative
SENTIMENT_SENSITIVE_INTENTS = {
    "refund_return",
    "damaged_defective",
    "complaint_feedback"
}

INTENT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "order_delivery": {
        "label": "Order Status & Delivery",
        "description": "Inquiries regarding package shipping, delivery delays, tracking, courier/driver issues, or packages marked delivered but not received.",
        "examples": [
            "My order was due to arrive yesterday but it hasn't even been dispatched yet? Order #408-4360134.",
            "Tracking says delivered to resident at 2pm but I was home and nothing arrived at my door.",
            "Delivery driver claimed they attempted delivery but nobody rang the bell."
        ]
    },
    "refund_return": {
        "label": "Refund & Return",
        "description": "Requests to return an item, tracking return pickup, queries about missing refund money, or cancelled orders awaiting refund.",
        "examples": [
            "My order was cancelled but I still have not received my refund in my bank account.",
            "Returned the item over a week ago, please confirm when my refund will be processed.",
            "How do I generate a return shipping label for my recent purchase?"
        ]
    },
    "damaged_defective": {
        "label": "Damaged, Defective or Wrong Item",
        "description": "Customer received a broken, tampered, defective item, missing components, or completely incorrect product.",
        "examples": [
            "Received my package today but the box was crushed and the bottle inside is shattered.",
            "I ordered a blue winter coat but received a red sweater instead.",
            "The package was opened and tampered with, and the second item is missing from the box."
        ]
    },
    "account_access": {
        "label": "Account Access & Security",
        "description": "Issues logging in, password reset problems, locked/suspended accounts, unauthorized login alerts, or 2FA verification issues.",
        "examples": [
            "My account has been locked for suspicious activity and I cannot log in to contact support.",
            "Someone hacked into my Amazon account and placed orders I didn't authorize.",
            "I'm not receiving the OTP code on my registered mobile number to reset my password."
        ]
    },
    "billing_payment": {
        "label": "Billing, Charges & Payments",
        "description": "Disputed charges, unauthorized debits, double charges, gift card balance holds, or payment gateway transaction failures.",
        "examples": [
            "I was charged twice on my credit card for a single order.",
            "There is an unauthorized charge of $79.99 on my bank statement from Amazon.",
            "My gift card balance was withheld and I cannot use my funds."
        ]
    },
    "subscription_prime": {
        "label": "Prime & Membership",
        "description": "Questions about Amazon Prime benefits, renewal fees, membership cancellation, Prime student, or delivery guarantees tied to Prime.",
        "examples": [
            "Why am I paying $139 a year for Prime if 2-day delivery is taking a week?",
            "How do I cancel my Amazon Prime auto-renewal before next month's charge?",
            "Does my Prime membership include Amazon Music Unlimited or do I pay extra?"
        ]
    },
    "technical_product": {
        "label": "Technical & Digital Products",
        "description": "Troubleshooting Amazon hardware/software: Echo, Alexa, Fire TV, Kindle e-readers, Kindle ebooks, Prime Video streaming bugs, or app crashes.",
        "examples": [
            "My Fire TV stick keeps freezing on the loading screen and rebooting.",
            "Alexa is not discovering my smart light bulbs in the app.",
            "Why can't I buy this Kindle edition book in my region?"
        ]
    },
    "complaint_feedback": {
        "label": "Complaint & General Feedback",
        "description": "General frustration, venting about poor service experience, praise/thanks, or feedback without a specific actionable transaction request.",
        "examples": [
            "Worst customer service experience I have ever had. Nobody knows what they are doing.",
            "Never buying from Amazon again, your customer service is completely incompetent.",
            "Huge thank you to representative Sarah on chat today, she solved my issue in 2 minutes!"
        ]
    }
}


def get_intent_prompt_block() -> str:
    """Formats taxonomy definitions and few-shot examples for LLM classification prompts."""
    lines = ["Available Intents:"]
    for intent_key, data in INTENT_DEFINITIONS.items():
        lines.append(f"\n- **{intent_key}** ({data['label']}): {data['description']}")
        lines.append("  Examples:")
        for ex in data["examples"]:
            lines.append(f"    * \"{ex}\"")
    return "\n".join(lines)
