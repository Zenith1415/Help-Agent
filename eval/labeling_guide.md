# Golden Evaluation Set Labeling Guide

This guide establishes the annotation standard for hand-labeling customer support tweets into our empirical 8-intent taxonomy and assigning ground-truth auto-handle vs. escalation decisions.

---

## 1. Intent Taxonomy & Annotation Rules

Every customer inquiry must be assigned to exactly one primary intent based on the core customer action request:

| Intent Key | Display Name | Core Scope | Escalation Default |
| :--- | :--- | :--- | :--- |
| `order_delivery` | Order Status & Delivery | Shipping delays, ETA, tracking links, courier complaints, lost/stolen packages. | Auto-handle (unless severe delay/frustration) |
| `refund_return` | Refund & Return | Return pickup status, return label, missing refund after return, order cancellation refund. | Escalate if disputed / negative sentiment |
| `damaged_defective` | Damaged / Defective / Wrong | Broken product, tampered seal, wrong size/color delivered, missing parts. | Escalate if replacement needed or angry |
| `account_access` | Account Access & Security | Locked account, hacked account, password reset failure, OTP issues, 2FA. | **Always Escalate (Hard Rule)** |
| `billing_payment` | Billing & Charges | Unauthorized charges, double debit, Amazon Pay balance withheld, payment failure. | **Always Escalate (Hard Rule)** |
| `subscription_prime` | Prime & Membership | Prime shipping speed complaints, renewal cancellation, membership fees, Prime benefits. | Auto-handle policy; escalate refunds |
| `technical_product` | Technical & Digital | Echo, Alexa, Fire TV, Kindle device/app errors, Prime video playback bugs. | Auto-handle troubleshooting steps |
| `complaint_feedback` | Complaint & Feedback | General brand venting, compliments, sarcasm, service feedback without an active order ask. | Escalate if hostile/abusive |

---

## 2. Disambiguation Rules (Edge Cases)

When a customer tweet touches multiple topics, follow these tie-breaker rules:
1. **Security & Financial Precedence**: If a message mentions an order problem AND an account security or unauthorized billing issue, prioritize `account_access` or `billing_payment`.
2. **Damaged vs. Delivery**: If an item arrived on time but broken, classify as `damaged_defective`. If an item did not arrive at all, classify as `order_delivery`.
3. **Refund vs. Delivery**: If the customer specifically demands "give me my money back / refund me" because an order is late, classify as `refund_return`. If they ask "where is my package", classify as `order_delivery`.
4. **Prime Speed Complaint vs. Prime Membership**: If the user asks why they pay for Prime when delivery is delayed, classify as `subscription_prime` if the inquiry challenges the membership value; classify as `order_delivery` if asking for an update on a specific delayed tracking number.

---

## 3. Escalation Decision Rubric

The ground-truth `should_escalate` label reflects whether an autonomous AI agent can safely complete the interaction or if human intervention is required:

### MUST Escalate (`should_escalate: true`):
- **Account Security**: Password resets, locked accounts, suspected identity theft, hacked accounts (requires human identity verification).
- **Financial Disputes**: Unauthorized transactions, credit card double billing, withheld balances (requires sensitive payment gateway lookup).
- **High Negative Sentiment / Legal Threats**: Hostile messages, threats of consumer court / legal action, severe brand damage.
- **Complex Exception Handling**: Tampered shipments, stolen packages requiring carrier claims or police reports.

### Can Auto-Handle (`should_escalate: false`):
- **Standard Order Inquiries**: Providing tracking links, carrier contact information, standard delivery time windows.
- **Self-Service Procedures**: How to return an item, where to download an invoice, how to cancel Prime auto-renewal.
- **Technical Troubleshooting**: Restarting devices, re-linking Alexa skills, clearing app cache.
- **Neutral / Positive Feedback**: Thanking the customer for kind words or acknowledging general service suggestions.
