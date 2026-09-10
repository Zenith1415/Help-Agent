# Help-Agent

AI Customer Support Agent for **AmazonHelp** built on real-world customer support conversations on Twitter.

## Overview

Help-Agent is an automated triage and response system designed to handle high-volume e-commerce customer support requests while maintaining safety and trust through an explicit escalation gate.

The system performs three core functions:
1. **Triage (Intent Classification)**: Categorizes incoming customer messages into a data-derived taxonomy of intents.
2. **Copilot (Grounded Reply Drafting)**: Drafts responses grounded in historical resolutions using FAISS retrieval over real brand-customer conversation pairs.
3. **Agent (Escalation Gate)**: Decides whether an incoming query can be safely auto-handled or must be escalated to a human agent, providing an explicit, auditable reason for every decision.

## Architecture

```
[ Incoming Tweet ]
        │
        ▼
[ 1. Triage / Intent Classifier ] ──► (Predicted Intent + Confidence)
        │
        ▼
[ 2. Escalation Gate ]
   ├─ Hard security/financial rules (Account / Billing)
   ├─ Negative sentiment + risk intents (Refund / Damaged / Complaint)
   └─ Low classifier confidence
        │
        ├── Escalate to Human (with logged reason)
        │
        └── Auto-handle
                │
                ▼
[ 3. Copilot / Reply Drafter ]
   ├─ FAISS vector search (top-3 historical pairs)
   └─ RAG-grounded LLM reply generation
```

## Repository Structure

```
Help-Agent/
├── data/                  # Data acquisition, filtering, and clustering
│   ├── download_data.py   # Kaggle dataset fetcher
│   ├── filter_brand.py    # Filter AmazonHelp multi-turn pairs
│   └── cluster_intents.py # Semantic clustering to derive taxonomy
├── pipeline/              # Core pipeline implementation
│   ├── intents.py         # Intent taxonomy & canonical examples
│   ├── classify.py        # Intent classification models
│   ├── retrieve.py        # FAISS vector store & retrieval
│   ├── draft.py           # RAG reply generation
│   ├── escalate.py        # Escalation policy with reason logging
│   └── run_pipeline.py    # End-to-end pipeline execution
├── baselines/             # Comparison baselines
│   ├── trivial.py         # Majority class & canned replies
│   └── simple.py          # TF-IDF + Logistic Regression & template replies
├── eval/                  # Evaluation harness
│   ├── golden_set.jsonl   # 150-250 hand-labeled golden test cases
│   ├── labeling_guide.md  # Annotation guidelines
│   ├── judge.py           # LLM-as-judge scoring rubric
│   ├── agreement_check.py # Human vs. judge agreement (Cohen's kappa)
│   ├── metrics.py         # Accuracy, Macro-F1, Precision/Recall
│   └── run_eval.py        # Automated headline evaluation runner
├── docs/
│   ├── report.md          # Technical report & findings
│   ├── decision_log.md    # Architecture & methodology decisions
│   └── CITATIONS.md       # External references & attributions
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup & Quickstart

### Prerequisites
- Python 3.11+
- Kaggle API token (`kaggle.json` or `KAGGLE_USERNAME` & `KAGGLE_KEY`)
- Anthropic API key (`ANTHROPIC_API_KEY`)

### Installation

```bash
# Clone the repository
git clone https://github.com/Zenith1415/Help-Agent.git
cd Help-Agent

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Reproducing Headline Results

Run the full evaluation pipeline in under 15 minutes:

```bash
python eval/run_eval.py
```
