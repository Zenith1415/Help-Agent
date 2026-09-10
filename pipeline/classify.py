"""
pipeline/classify.py
Main Intent Classifier using Google Gemini Few-Shot Prompting.
Prompt incorporates empirical taxonomy definitions and canonical examples.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Optional
from pipeline.llm import get_llm_client
from pipeline.intents import INTENTS, get_intent_prompt_block

SYSTEM_PROMPT = """You are an expert customer support triage AI for AmazonHelp on Twitter.
Your job is to classify incoming customer inquiries into exactly ONE intent category from the provided taxonomy.
You must return your response in JSON format with three fields:
1. "intent": string, exactly one of the allowed intent keys.
2. "confidence": float between 0.0 and 1.0 indicating your certainty.
3. "rationale": brief explanation for why this intent was chosen.
"""


def build_classification_prompt(customer_msg: str) -> str:
    prompt = f"""Review the following customer message sent to AmazonHelp and classify its primary intent.

{get_intent_prompt_block()}

Allowed Intent Keys:
{', '.join(INTENTS)}

Customer Message:
"{customer_msg}"

Respond strictly with a JSON object:
{{
  "intent": "<intent_key>",
  "confidence": <float between 0.0 and 1.0>,
  "rationale": "<one line explanation>"
}}
"""
    return prompt


class IntentClassifier:
    """Few-shot LLM intent classifier with fallback to simple baseline."""

    def __init__(self, fallback_to_baseline: bool = True):
        self.client = get_llm_client()
        self.fallback_to_baseline = fallback_to_baseline
        self._fallback_model = None

    def _get_fallback(self):
        if self._fallback_model is None:
            from baselines.simple import SimpleBaseline
            self._fallback_model = SimpleBaseline()
        return self._fallback_model

    def classify(self, customer_msg: str) -> Dict[str, Any]:
        """Classify customer message into intent with confidence score."""
        prompt = build_classification_prompt(customer_msg)

        try:
            res = self.client.generate_json(
                prompt=prompt,
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0
            )

            intent = res.get("intent", "").strip().lower()
            confidence = float(res.get("confidence", 0.85))
            rationale = res.get("rationale", "")

            # Ensure valid intent
            if intent not in INTENTS:
                # Attempt to find closest match
                for valid_k in INTENTS:
                    if valid_k in intent or intent in valid_k:
                        intent = valid_k
                        break
                else:
                    intent = "order_delivery"

            return {
                "intent": intent,
                "confidence": min(max(confidence, 0.0), 1.0),
                "rationale": rationale,
                "model": "gemini_few_shot"
            }

        except Exception as e:
            if self.fallback_to_baseline:
                fallback = self._get_fallback()
                fb_res = fallback.classify_intent(customer_msg)
                fb_res["rationale"] = f"Fallback due to LLM error: {e}"
                return fb_res
            raise


# Global singleton instance
_classifier_instance = None

def get_intent_classifier() -> IntentClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance
