"""
pipeline/draft.py
Grounded Reply Generator (Copilot) using FAISS Retrieval and LLM RAG Prompting.
Enforces tone matching, factual grounding, and avoids hallucinating policy promises.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Dict, Any, Optional
from pipeline.llm import get_llm_client

DRAFT_SYSTEM_PROMPT = """You are an AI Copilot drafting customer service responses for AmazonHelp on Twitter.
Your goal is to draft a helpful, professional, and empathetic reply that matches AmazonHelp's historical communication style.

STRICT GROUNDING RULES:
1. Tone: Polite, concise, solution-oriented, Twitter-appropriate (< 280 characters if possible).
2. Grounding: Ground your suggested next steps in how AmazonHelp has historically resolved similar issues.
3. No Hallucinations: DO NOT invent specific refund amounts, guarantee delivery dates, or make policy promises not supported by the context.
4. If personal account details or secure authentication are needed, advise the customer to reach out via their official account page or secure customer service link.
5. Return ONLY the drafted tweet text. Do not add quotes, commentary, or markdown formatting around the reply.
"""


def format_rag_prompt(customer_msg: str, intent: str, retrieved_pairs: List[Dict[str, Any]]) -> str:
    examples_block = []
    for i, pair in enumerate(retrieved_pairs, 1):
        examples_block.append(
            f"Example {i}:\n"
            f"Customer: \"{pair.get('customer_msg', '')}\"\n"
            f"AmazonHelp Reply: \"{pair.get('brand_reply', '')}\""
        )
    examples_text = "\n\n".join(examples_block) if examples_block else "No specific examples found."

    prompt = f"""Historical AmazonHelp resolutions for similar issues:
{examples_text}

---
Incoming Customer Message:
Intent: {intent}
Customer: "{customer_msg}"

Draft a grounded reply to this customer adhering strictly to the grounding rules:"""
    return prompt


class ReplyDrafter:
    """RAG-grounded customer service reply drafter."""

    def __init__(self):
        self.client = get_llm_client()

    def draft(
        self,
        customer_msg: str,
        intent: str,
        retrieved_pairs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Draft a grounded response using retrieved historical pairs."""
        prompt = format_rag_prompt(customer_msg, intent, retrieved_pairs)

        try:
            draft_text = self.client.generate(
                prompt=prompt,
                system_instruction=DRAFT_SYSTEM_PROMPT,
                temperature=0.2
            )
            # Strip quotes if returned
            draft_text = draft_text.strip().strip('"').strip("'")
            return {
                "draft_reply": draft_text,
                "retrieved_context": retrieved_pairs,
                "model": "gemini_rag_draft"
            }
        except Exception as e:
            # Fallback to simple template reply
            from baselines.simple import TEMPLATES
            fallback_text = TEMPLATES.get(intent, TEMPLATES["order_delivery"])
            return {
                "draft_reply": fallback_text,
                "retrieved_context": retrieved_pairs,
                "model": f"fallback_template ({e})"
            }


_drafter_instance = None

def get_reply_drafter() -> ReplyDrafter:
    global _drafter_instance
    if _drafter_instance is None:
        _drafter_instance = ReplyDrafter()
    return _drafter_instance
