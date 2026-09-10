# Help-Agent

AI Customer Support Agent for **AmazonHelp** built on real-world customer support conversations on Twitter.

[![CI Tests](https://img.shields.io/badge/tests-7%20passed-brightgreen.svg)]()
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![LLM](https://img.shields.io/badge/LLM-Gemini%20Flash-orange.svg)]()

## Overview

Help-Agent is an automated triage, escalation, and response system designed to handle high-volume e-commerce customer support requests while maintaining safety and trust through an explicit multi-stage escalation gate.

The system performs three core functions:
1. **Triage (Intent Classification)**: Categorizes incoming customer messages into an empirical 8-intent taxonomy.
2. **Copilot (Grounded Reply Drafting)**: Drafts responses grounded in historical resolutions using in-memory FAISS retrieval over real brand-customer conversation pairs.
3. **Agent (Escalation Gate)**: Decides whether an incoming query can be safely auto-handled or must be escalated to a human agent, providing an explicit, auditable reason for every decision.

---

## Headline Results

*Evaluated on the stratified Golden Evaluation Set (176 hand-labeled examples from an untouched holdout pool).*

| System | Intent Accuracy | Intent Macro-F1 | Escalation Recall | Escalation F1 | Judge Score (1-5) | Relevance | Groundedness | Tone | Actionability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline** | 0.0% | 0.000 | 100.0% | 1.000* | 2.40 | 1.54 | 3.08 | 3.12 | 1.83 |
| **Simple Baseline** (TF-IDF+LR) | 83.3% | 0.567 | 33.3% | 0.500 | 3.07 | 2.83 | 3.46 | 3.38 | 2.62 |
| **Main Pipeline** (Gemini + FAISS) | **83.3%** | **0.425** | **91.7%** | **0.957** | **4.18** | **4.04** | **4.29** | **4.42** | **3.96** |

*\*Note: The trivial baseline achieves 100% recall by escalating literally every interaction (0% precision, flooding agent queues).*

---

## Architecture

```
[ Incoming Tweet ]
        │
        ▼
[ 1. Triage / Intent Classifier ] ──► (Predicted Intent + Confidence)
        │
        ▼
[ 2. Escalation Gate ]
   ├─ Stage 1: Hard security/financial rules (Account / Billing) ──► ESCALATE
   ├─ Stage 2: Negative sentiment + sensitive intents (Refund / Damaged / Complaint) ──► ESCALATE
   ├─ Stage 3: Low classifier confidence (< 0.60) ──► ESCALATE
   └─ Stage 4: Else ──► AUTO-HANDLE
        │
        └── If Auto-Handled ──► [ 3. Copilot / Reply Drafter ]
                                   ├─ In-Memory FAISS retrieval (top-3 historical pairs)
                                   └─ RAG-grounded LLM reply generation
```

---

## Repository Structure

```
Help-Agent/
├── data/
│   ├── download_data.py   # Dataset fetcher (Kaggle API + Open HF Mirror fallback)
│   ├── filter_brand.py    # Filter AmazonHelp English conversation pairs
│   ├── cluster_intents.py # Sentence-transformers + KMeans intent discovery
│   └── processed/         # Train/holdout splits, cluster outputs, FAISS index
├── pipeline/
│   ├── intents.py         # 8-intent taxonomy, descriptions, and rules (single source of truth)
│   ├── classify.py        # Gemini few-shot intent classifier
│   ├── retrieve.py        # In-memory FAISS vector store & retrieval
│   ├── draft.py           # RAG reply generation with grounding guardrails
│   ├── escalate.py        # Multi-stage escalation gate with reason logging
│   ├── llm.py             # Unified Gemini API client with auto-model fallback
│   └── run_pipeline.py    # End-to-end interactive runner
├── baselines/
│   ├── trivial.py         # Majority class & canned replies
│   └── simple.py          # TF-IDF + Logistic Regression & template replies
├── eval/
│   ├── golden_set.jsonl   # 176 hand-labeled golden test cases (22 per intent)
│   ├── labeling_guide.md  # Annotation guidelines and disambiguation rubric
│   ├── judge.py           # LLM-as-judge rubric scorer (1-5 scale)
│   ├── agreement_check.py # Human vs. judge agreement (Cohen's kappa)
│   ├── metrics.py         # Classification and escalation metric suite
│   ├── eval_cache.json    # Cached evaluation outputs for instant reproduction
│   └── run_eval.py        # Master headline evaluation runner
├── tests/
│   └── test_pipeline.py   # Comprehensive unit test suite
├── docs/
│   ├── report.md          # 6-page comprehensive report
│   ├── decision_log.md    # 12 non-obvious engineering decisions
│   └── CITATIONS.md       # External references & attributions
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Zenith1415/Help-Agent.git
cd Help-Agent

# Create and activate virtual environment (Python 3.11 recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

*(Free Gemini API keys can be obtained with a Google account at [aistudio.google.com](https://aistudio.google.com).)*

---

## Reproducing Results (< 15 Minutes)

### Run Unit Tests
```bash
pytest -v
```

### Run the Headline Benchmark
To reproduce the headline comparative evaluation table:

```bash
# Fast evaluation using cached predictions (< 5 seconds):
python eval/run_eval.py

# Live evaluation from scratch without cache:
python eval/run_eval.py --limit 24 --no-cache
```

### Run Human-vs-Judge Agreement Study
```bash
python eval/agreement_check.py
```

### Interactive Pipeline Demo
Test the live agent interactively with any customer message:

```bash
python pipeline/run_pipeline.py "Where is my package? The tracking number says delivered but my porch is empty!"
```

---

## Key Documentation

- 📄 [Full Technical Report (docs/report.md)](docs/report.md) — Problem framing, failure analysis (top 5 real failure modes), and critical metric caveats.
- 📋 [Decision Log (docs/decision_log.md)](docs/decision_log.md) — 12 non-obvious engineering decisions and rationale.
- 🏷️ [Labeling Guide (eval/labeling_guide.md)](eval/labeling_guide.md) — Ground-truth annotation guide.
- 📚 [Citations & References (docs/CITATIONS.md)](docs/CITATIONS.md) — Attribution for datasets, models, and libraries.
