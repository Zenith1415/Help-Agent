"""
tests/test_pipeline.py
Unit tests for Help-Agent core components:
- Intent taxonomy integrity
- Escalation gate rules & cascades
- Baseline models
- Evaluation metrics
"""

import pytest
from pipeline.intents import (
    INTENTS,
    HARD_ESCALATE_INTENTS,
    SENTIMENT_SENSITIVE_INTENTS,
    INTENT_DEFINITIONS
)
from pipeline.escalate import EscalationGate
from baselines.trivial import TrivialBaseline
from baselines.simple import SimpleBaseline
from eval.metrics import evaluate_intent_classification, evaluate_escalation_gate


def test_intent_taxonomy_integrity():
    """Verify all 8 intents are defined with descriptions and canonical examples."""
    assert len(INTENTS) == 8
    for intent in INTENTS:
        assert intent in INTENT_DEFINITIONS
        info = INTENT_DEFINITIONS[intent]
        assert "label" in info and len(info["label"]) > 0
        assert "description" in info and len(info["description"]) > 10
        assert "examples" in info and len(info["examples"]) >= 2


def test_escalation_gate_hard_rules():
    """Verify account_access and billing_payment always escalate."""
    gate = EscalationGate()
    
    # Account access
    res_acc = gate.evaluate("I forgot my password and cannot sign in", "account_access", 0.95)
    assert res_acc["should_escalate"] is True
    assert "hard rule" in res_acc["reason"]
    assert res_acc["stage"] == 1

    # Billing payment
    res_bill = gate.evaluate("Why was I billed twice on my card?", "billing_payment", 0.90)
    assert res_bill["should_escalate"] is True
    assert "hard rule" in res_bill["reason"]
    assert res_bill["stage"] == 1


def test_escalation_gate_sentiment_cascade():
    """Verify negative sentiment escalates on sensitive intents."""
    gate = EscalationGate()

    # Negative sentiment on refund
    res_neg_refund = gate.evaluate("Terrible horrible service, refund my money immediately you scam!", "refund_return", 0.85)
    assert res_neg_refund["should_escalate"] is True
    assert res_neg_refund["stage"] == 2

    # Neutral sentiment on order delivery
    res_neutral = gate.evaluate("Hello, could you please give me tracking info for order 123?", "order_delivery", 0.90)
    assert res_neutral["should_escalate"] is False
    assert res_neutral["stage"] == 4


def test_escalation_gate_low_confidence():
    """Verify low model confidence triggers escalation."""
    gate = EscalationGate(confidence_threshold=0.60)
    res_low = gate.evaluate("Maybe it was sent or not", "order_delivery", 0.45)
    assert res_low["should_escalate"] is True
    assert res_low["stage"] == 3


def test_trivial_baseline():
    """Verify trivial baseline predicts majority class and always escalates."""
    tb = TrivialBaseline()
    cls_res = tb.classify_intent("Any message")
    assert cls_res["intent"] == "order_delivery"
    esc_res = tb.decide_escalation("Any message", cls_res["intent"])
    assert esc_res["should_escalate"] is True


def test_simple_baseline():
    """Verify simple baseline classifies and outputs valid schema."""
    sb = SimpleBaseline()
    cls_res = sb.classify_intent("My package has not been delivered yet")
    assert cls_res["intent"] in INTENTS
    assert 0.0 <= cls_res["confidence"] <= 1.0


def test_evaluation_metrics():
    """Verify metrics computation."""
    y_true = ["order_delivery", "refund_return", "account_access"]
    y_pred = ["order_delivery", "refund_return", "billing_payment"]
    res = evaluate_intent_classification(y_true, y_pred)
    assert "accuracy" in res
    assert "macro_f1" in res

    y_esc_true = [True, False, True]
    y_esc_pred = [True, False, False]
    esc_res = evaluate_escalation_gate(y_esc_true, y_esc_pred)
    assert "precision" in esc_res
    assert "recall" in esc_res
    assert esc_res["recall"] == 0.5
