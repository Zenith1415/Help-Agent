# Headline Benchmark Results

*Evaluated on 24 stratified golden set examples. Execution time: 0.0s.*

| System | Intent Accuracy | Intent Macro-F1 | Escalation Recall | Escalation F1 | Judge Composite (1-5) | Relevance | Groundedness | Tone | Actionability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline** | 0.0% | 0.000 | 100.0% | 1.000 | 2.40 | 1.54 | 3.08 | 3.12 | 1.83 |
| **Simple Baseline**  | 83.3% | 0.567 | 33.3% | 0.500 | 3.07 | 2.83 | 3.46 | 3.38 | 2.62 |
| **Main Pipeline**    | **83.3%** | **0.425** | **91.7%** | **0.957** | **4.18** | **4.04** | **4.29** | **4.42** | **3.96** |
