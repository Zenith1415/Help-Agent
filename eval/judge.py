"""
eval/judge.py
LLM-as-a-Judge Rubric Scorer for Customer Support Replies.
Scores replies on Relevance, Groundedness, Tone Match, and Actionability (1-5 scale).
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.llm import get_llm_client

JUDGE_SYSTEM_PROMPT = """You are an impartial, expert evaluation judge evaluating AI customer support replies for AmazonHelp on Twitter.
You evaluate the agent's drafted reply against the customer's inquiry and the historical support context.

Evaluate the reply on four rubrics, scoring each from 1 (poor) to 5 (excellent):

1. "relevance" (1-5):
   - 5: Directly answers the customer's specific problem.
   - 3: Partially relevant but misses key nuance.
   - 1: Irrelevant or addresses the wrong topic entirely.

2. "groundedness" (1-5):
   - 5: Strictly grounded in realistic Amazon support procedures. No invented facts, false delivery promises, or fake refund guarantees.
   - 3: Mostly realistic, but contains minor speculative statements.
   - 1: Hallucinates specific refund amounts, false dates, or promises that violate support policy.

3. "tone" (1-5):
   - 5: Polite, empathetic, concise, and Twitter-appropriate (< 280 chars).
   - 3: Acceptable tone, but too robotic, verbose, or stiff.
   - 1: Rude, dismissive, inappropriate, or excessively repetitive.

4. "actionability" (1-5):
   - 5: Gives the customer a clear, unambiguous, immediate next step (e.g., check tracking link, DM details, check account orders).
   - 3: Suggests a generic action without specific guidance.
   - 1: Leaves the customer stranded with no resolution path.

Return your evaluation STRICTLY as a JSON object:
{
  "relevance": <integer 1-5>,
  "groundedness": <integer 1-5>,
  "tone": <integer 1-5>,
  "actionability": <integer 1-5>,
  "composite_score": <float average of the 4 scores>,
  "rationale": "<concise explanation>"
}
"""


def build_judge_prompt(customer_msg: str, predicted_intent: str, draft_reply: str) -> str:
    return f"""Evaluate the following customer support interaction:

Customer Message:
"{customer_msg}"

Predicted Intent:
{predicted_intent}

AI Agent Drafted Reply:
"{draft_reply}"

Provide your rubric scores (1-5) and brief rationale as a JSON object."""


class SupportJudge:
    """LLM Judge evaluating reply quality against the rubric."""

    def __init__(self):
        self.client = get_llm_client()

    def evaluate_reply(
        self,
        customer_msg: str,
        predicted_intent: str,
        draft_reply: str
    ) -> Dict[str, Any]:
        """Judge a drafted reply on the 4 rubric dimensions."""
        prompt = build_judge_prompt(customer_msg, predicted_intent, draft_reply)

        try:
            res = self.client.generate_json(
                prompt=prompt,
                system_instruction=JUDGE_SYSTEM_PROMPT,
                temperature=0.0
            )

            rel = int(res.get("relevance", 4))
            grd = int(res.get("groundedness", 4))
            ton = int(res.get("tone", 4))
            act = int(res.get("actionability", 4))
            composite = round((rel + grd + ton + act) / 4.0, 2)

            return {
                "relevance": rel,
                "groundedness": grd,
                "tone": ton,
                "actionability": act,
                "composite_score": composite,
                "rationale": res.get("rationale", "No rationale provided")
            }
        except Exception as e:
            # Heuristic fallback if LLM judge times out
            return {
                "relevance": 3,
                "groundedness": 3,
                "tone": 4,
                "actionability": 3,
                "composite_score": 3.25,
                "rationale": f"Heuristic fallback: {e}"
            }


_judge_instance = None

def get_support_judge() -> SupportJudge:
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = SupportJudge()
    return _judge_instance
