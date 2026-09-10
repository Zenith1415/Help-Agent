# Headline Benchmark Results

*Evaluated on 24 stratified golden set examples. Execution time: 244.3s.*

| System | Intent Accuracy | Intent Macro-F1 | Escalation Recall | Escalation F1 | Judge Composite (1-5) | Relevance | Groundedness | Tone | Actionability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline** | 0.0% | 0.000 | 100.0% | 1.000 | 2.40 | 1.42 | 3.25 | 3.04 | 1.88 |
| **Simple Baseline**  | 83.3% | 0.567 | 33.3% | 0.500 | 3.12 | 2.88 | 3.58 | 3.38 | 2.67 |
| **Main Pipeline**    | **87.5%** | **0.576** | **91.7%** | **0.957** | **4.04** | **3.92** | **4.42** | **4.08** | **3.75** |
