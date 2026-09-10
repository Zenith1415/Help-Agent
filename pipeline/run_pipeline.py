"""
pipeline/run_pipeline.py
End-to-End Execution Pipeline:
Customer Tweet -> Intent Classification -> Escalation Gate -> FAISS Retrieval -> RAG Draft Reply.
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from pipeline.classify import get_intent_classifier
from pipeline.escalate import get_escalation_gate
from pipeline.retrieve import get_retriever
from pipeline.draft import get_reply_drafter


def process_message(customer_msg: str) -> dict:
    """Execute full support agent pipeline on an incoming customer inquiry."""
    # 1. Triage: Intent Classification
    classifier = get_intent_classifier()
    classification = classifier.classify(customer_msg)
    intent = classification["intent"]
    confidence = classification["confidence"]

    # 2. Agent: Escalation Decision
    gate = get_escalation_gate()
    escalation = gate.evaluate(customer_msg, intent, confidence)

    # 3. Retrieval: Similar Historical Pairs
    retriever = get_retriever()
    retrieved_pairs = retriever.retrieve(customer_msg, top_k=3)

    # 4. Copilot: Grounded Reply Drafting
    drafter = get_reply_drafter()
    draft_result = drafter.draft(customer_msg, intent, retrieved_pairs)

    return {
        "customer_message": customer_msg,
        "classification": classification,
        "escalation": escalation,
        "draft_reply": draft_result["draft_reply"],
        "retrieved_context_count": len(retrieved_pairs),
        "status": "escalated_to_human" if escalation["should_escalate"] else "auto_handled"
    }


def main():
    parser = argparse.ArgumentParser(description="Run Help-Agent pipeline on a customer tweet.")
    parser.add_argument("message", type=str, nargs="?", default="Where is my package? The tracking number says delivered but my porch is empty!", help="Incoming customer tweet")
    parser.add_argument("--json", action="store_true", help="Print output strictly as JSON")
    args = parser.parse_args()

    result = process_message(args.message)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("\n=== Help-Agent Triage & Response ===")
        print(f"Customer Tweet   : \"{result['customer_message']}\"")
        print(f"Predicted Intent : {result['classification']['intent']} (confidence: {result['classification']['confidence']:.2f})")
        print(f"Classification   : {result['classification'].get('rationale', 'N/A')}")
        print(f"Escalation Gate  : {'[ESCALATE TO HUMAN]' if result['escalation']['should_escalate'] else '[AUTO-HANDLE]'}")
        print(f"Escalation Reason: {result['escalation']['reason']}")
        print(f"Drafted Response : \"{result['draft_reply']}\"")
        print(f"Grounded Context : {result['retrieved_context_count']} historical pairs retrieved\n")


if __name__ == "__main__":
    main()
