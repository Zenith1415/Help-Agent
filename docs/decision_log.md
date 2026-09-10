# Architecture & Implementation Decision Log

This document records 12 key non-obvious technical and architectural decisions made throughout the development of **Help-Agent**, providing explicit rationale for why specific approaches were selected over alternatives.

---

### 1. Brand Selection: AmazonHelp
- **Decision**: Focus exclusively on `@AmazonHelp` from the 3M-tweet Kaggle dataset.
- **Rationale**: Amazon customer support represents high-volume, diverse e-commerce inquiries (delivery delays, refunds, damaged goods, account issues). These problem types map directly to real-world support desk operations (like Hiver's core customer base) without requiring deep brand-specific proprietary technical knowledge to understand and evaluate.

### 2. LLM Provider: Google Gemini API (Flash series)
- **Decision**: Switched from Anthropic Claude API to Google Gemini API (`gemini-3.6-flash`).
- **Rationale**: Gemini offers a generous free tier (no credit card barrier), fast token throughput, reliable JSON structured outputs for classification, and low latency for RAG drafting and LLM-as-judge evaluation.

### 3. Embeddings & Retrieval: `all-MiniLM-L6-v2` + In-Memory FAISS
- **Decision**: Use in-memory FAISS with sentence-transformers `all-MiniLM-L6-v2` rather than managed vector databases (Pinecone, Qdrant) or heavy LLM embeddings.
- **Rationale**: At the scale of tens of thousands of historical support pairs, `all-MiniLM-L6-v2` runs efficiently on standard CPU without GPU acceleration, maintains high semantic similarity quality for short text tweets, and FAISS in-memory index provides sub-millisecond retrieval with zero external infrastructure overhead.

### 4. Intent Granularity: 8 Empirical Intents vs. 77+ (Banking77)
- **Decision**: Collapse customer queries into 8 primary operational intents derived via KMeans clustering rather than an exhaustive multi-level taxonomy.
- **Rationale**: High granularity (e.g. Banking77's 77 classes) leads to boundary blur and severe label confusion on short social media posts. 8 operational categories map directly to distinct downstream agent routing: hard escalation (Account, Billing), sentiment-sensitive escalation (Refund, Damaged, Complaint), and automated resolution (Order status, Product queries).

### 5. Multi-Turn Conversation Turn Extraction: First Inbound & First Reply
- **Decision**: Pair each initial customer tweet strictly with the brand agent's immediate first response, ignoring tertiary follow-up loops.
- **Rationale**: In social support, the initial customer message contains the core problem statement before context is fragmented across multiple follow-ups. Grounding the reply generator on first-response brand actions mirrors real-world support triage and prevents topic dilution.

### 6. Escalation Gate as a Prioritized Multi-Stage Cascade
- **Decision**: Prioritize hard policy rules first, followed by sentiment + intent risk combinations, followed by classifier confidence thresholds.
- **Rationale**: Pure machine learning classification confidence is insufficient for mission-critical risk management. Security issues (unauthorized access) and billing disputes must never be auto-handled regardless of model confidence score.

### 7. VADER Sentiment Analysis as a Secondary Escalation Trigger
- **Decision**: Use VADER sentiment analysis strictly as a secondary gate modifier rather than a primary classification signal.
- **Rationale**: Sentiment alone does not determine support actionability (a polite customer may still have an unauthorized credit card charge, while a frustrated customer may simply need a tracking link). Coupling sentiment only with high-risk intents (refunds, damaged goods, complaints) prevents over-escalating benign inquiries.

### 8. Strict Holdout Pool Partitioning on Day 1
- **Decision**: Carve out 10,000 pairs into `holdout_pool.csv` before building the FAISS retrieval index or training baseline models.
- **Rationale**: Prevents data leakage where golden evaluation examples could accidentally appear in the retrieval database or classifier few-shot examples, ensuring all benchmark numbers represent true out-of-distribution generalization.

### 9. Filter Boilerplate "Send us a DM" Responses from Retrieval Index
- **Decision**: Demote and filter historical brand replies that contain only generic redirects like "Please send us a DM".
- **Rationale**: Over 40% of Twitter customer service replies are generic boilerplate redirects. If retrieved as RAG context, the LLM simply mimics the non-informative canned phrase. Filtering for substantive resolutions produces far more actionable drafted replies.

### 10. Stratified Golden Set Sampling (22 per intent)
- **Decision**: Sample an equal number of golden evaluation examples across all 8 intents rather than matching the natural power-law distribution.
- **Rationale**: High-risk intents like `account_access` (4.7%) and `billing_payment` (9.2%) are low-volume in raw data. Natural sampling would yield too few instances to reliably measure escalation recall on critical security failure modes.

### 11. Dual-Metric Escalation Evaluation: Prioritizing Recall over Precision
- **Decision**: Benchmark the escalation gate with a primary objective of maximizing Recall ($\ge 90\%$) while maintaining acceptable Precision ($\ge 75\%$).
- **Rationale**: In customer support operations, a False Negative (failing to escalate a hacked account or fraudulent billing charge) causes catastrophic customer churn or legal liability. A False Positive (escalating a question that could have been automated) only costs human handling time.

### 12. Transparent Reporting of Cohen's Kappa Agreement
- **Decision**: Report Cohen's kappa for LLM judge vs. human evaluation honestly ($0.183$), acknowledging that exact match ($45.0\%$) reflects natural subjectivity and systematic leniency in automated LLM evaluation.
- **Rationale**: Pretending that automated LLM evaluation perfectly mirrors human consensus is dishonest. Documenting where human and model judgments diverge (e.g. LLM judge's leniency on polite boilerplate vs. human annotator demands for specific resolution data) establishes authentic credibility for the evaluation harness.
