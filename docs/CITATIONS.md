# Citations and Attributions

This document logs all external libraries, datasets, models, prompt patterns, and templates borrowed or referenced during the development of **Help-Agent**, as required by project guidelines.

---

## 1. Datasets

### Customer Support on Twitter
- **Author/Source**: Thought Vector on Kaggle (`thoughtvector/customer-support-on-twitter`)
- **URL**: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- **Description**: ~3 million customer service tweets and brand replies across major companies, including `@AmazonHelp`.
- **Usage**: Source corpus for reconstructing historical customer-support pairs for training, clustering, retrieval, and evaluation.

### Banking77 (Reference Taxonomy)
- **Author/Source**: PolyAI (`PolyAI/banking77`) on HuggingFace
- **URL**: https://huggingface.co/datasets/PolyAI/banking77
- **Usage**: Used strictly as an architectural reference for intent design granularity vs. customer service domain scope.

---

## 2. Models & Pretrained Weights

### Sentence-Transformers / all-MiniLM-L6-v2
- **Authors**: Nils Reimers and Iryna Gurevych (UKP Lab)
- **Model Card**: `sentence-transformers/all-MiniLM-L6-v2`
- **URL**: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **Usage**: In-memory dense embeddings for customer message intent clustering and FAISS vector retrieval.

### Google Gemini API (`gemini-2.0-flash` / `gemini-1.5-flash`)
- **Provider**: Google DeepMind / Google AI Studio
- **Usage**: Few-shot intent classification, grounded RAG response generation, and LLM-as-judge rubric evaluation.

### VADER Sentiment Analysis
- **Authors**: C.J. Hutto and Eric Gilbert
- **Paper**: *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text* (ICWSM 2014)
- **Usage**: Secondary rule in escalation gate to detect frustrated/negative customer sentiment.

---

## 3. Libraries & Tools

- **FAISS (Facebook AI Similarity Search)**: Meta Research (`faiss-cpu`) for fast dense vector nearest-neighbor search.
- **Scikit-Learn**: Pedregosa et al. for KMeans clustering, TF-IDF vectorization, Logistic Regression baseline, and evaluation metrics (Cohen's kappa, Macro-F1).
- **Pandas / NumPy**: Wes McKinney et al. for tabular data transformation and pair reconstruction.

---

## 4. AI Assistance & Pair Programming Disclosure

In accordance with assignment guidelines encouraging transparent attribution of AI coding assistants:

- **Tools Used**: Google Antigravity / Gemini CLI coding assistant (powered by Gemini models).
- **Role of AI Assistant**:
  - Code scaffolding and boilerplate generation (fast chunked streaming in `data/filter_brand.py`, FAISS retrieval wrapper in `pipeline/retrieve.py`, unit test fixtures in `tests/test_pipeline.py`).
  - Evaluation metric implementations (`eval/metrics.py`, Cohen's kappa calculation in `eval/agreement_check.py`).
  - Initial drafting of technical documentation and markdown summaries.
- **Role of Human Author / Engineer**:
  - All high-level architectural decisions (selecting AmazonHelp, 8-intent operational granularity, in-memory FAISS vs fine-tuning, prioritizing escalation recall over precision).
  - Designing the multi-stage priority escalation cascade and explicit reason logging policies.
  - Curating and hand-labeling the 176-example golden evaluation set from the 10k holdout pool.
  - Identifying, diagnosing, and explaining the **Accuracy vs. Macro-F1 Paradox** and semantic divergence between surface keyword matching and pragmatic LLM understanding.
  - Conducting failure analysis and blind human-vs-judge scoring.
  - Comprehensive line-by-line code review, live debugging, and final approval of all system components. Ready to explain and modify any section of code live during review.
