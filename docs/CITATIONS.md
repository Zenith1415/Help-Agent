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
