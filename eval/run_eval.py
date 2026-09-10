"""
eval/run_eval.py
Master evaluation script that runs full comparative benchmark across:
1. Main Pipeline (Gemini Few-Shot + FAISS RAG + Escalation Cascade)
2. Simple Baseline (TF-IDF + Logistic Regression + Template + Threshold)
3. Trivial Baseline (Majority Class + Canned Reply + Always Escalate)

Produces headline results table in under 15 minutes (under 30 seconds with evaluation cache).
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from pipeline.run_pipeline import process_message
from baselines.trivial import TrivialBaseline
from baselines.simple import SimpleBaseline
from eval.metrics import evaluate_intent_classification, evaluate_escalation_gate
from eval.judge import get_support_judge

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_SET_FILE = EVAL_DIR / "golden_set.jsonl"
EVAL_CACHE_FILE = EVAL_DIR / "eval_cache.json"
HEADLINE_RESULTS_JSON = EVAL_DIR / "headline_results.json"
RESULTS_SUMMARY_MD = EVAL_DIR / "results_summary.md"


def load_cache() -> Dict[str, Any]:
    if EVAL_CACHE_FILE.exists():
        try:
            with open(EVAL_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache: Dict[str, Any]):
    try:
        with open(EVAL_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[!] Warning: Could not save cache: {e}")


def run_benchmark(sample_limit: int = None, run_judge: bool = True, use_cache: bool = True):
    """Run full evaluation on golden set examples across main system and baselines."""
    print("=" * 70)
    print("   Help-Agent: Headline Evaluation Harness (AmazonHelp Support)")
    print("=" * 70)

    if not GOLDEN_SET_FILE.exists():
        raise FileNotFoundError(f"Golden set file not found: {GOLDEN_SET_FILE}. Run eval/build_golden_set.py first.")

    with open(GOLDEN_SET_FILE, "r", encoding="utf-8") as f:
        golden_data = [json.loads(line) for line in f]

    if sample_limit:
        golden_data = golden_data[:sample_limit]
        print(f"[i] Running on subsample of {len(golden_data)} golden examples...")
    else:
        print(f"[i] Evaluating across golden set ({len(golden_data)} examples)...")

    cache = load_cache() if use_cache else {}
    cache_modified = False

    trivial = TrivialBaseline()
    simple = SimpleBaseline()
    judge = get_support_judge() if run_judge else None

    # Ground truth
    y_true_intent = [item["true_intent"] for item in golden_data]
    y_true_escalate = [item["should_escalate"] for item in golden_data]

    # Predictions storage
    preds = {
        "trivial": {"intent": [], "escalate": [], "replies": [], "judge_scores": []},
        "simple": {"intent": [], "escalate": [], "replies": [], "judge_scores": []},
        "main": {"intent": [], "escalate": [], "replies": [], "judge_scores": []}
    }

    start_time = time.time()

    # Step 1: Run Trivial Baseline
    print("\n[1/4] Running Trivial Baseline (Majority Class / Canned / Always Escalate)...")
    for item in golden_data:
        msg = item["customer_msg"]
        t_cls = trivial.classify_intent(msg)
        t_esc = trivial.decide_escalation(msg, t_cls["intent"])
        t_rep = trivial.draft_reply(msg, t_cls["intent"])
        preds["trivial"]["intent"].append(t_cls["intent"])
        preds["trivial"]["escalate"].append(t_esc["should_escalate"])
        preds["trivial"]["replies"].append(t_rep)

    # Step 2: Run Simple Baseline
    print("[2/4] Running Simple Baseline (TF-IDF + LR / Template / Threshold)...")
    for item in golden_data:
        msg = item["customer_msg"]
        s_cls = simple.classify_intent(msg)
        s_esc = simple.decide_escalation(msg, s_cls["intent"], s_cls["confidence"])
        s_rep = simple.draft_reply(msg, s_cls["intent"])
        preds["simple"]["intent"].append(s_cls["intent"])
        preds["simple"]["escalate"].append(s_esc["should_escalate"])
        preds["simple"]["replies"].append(s_rep)

    # Step 3: Run Main Pipeline
    print("[3/4] Running Main Pipeline (Gemini Few-Shot + Escalation Gate + FAISS RAG)...")
    for i, item in enumerate(golden_data, 1):
        item_id = item["id"]
        msg = item["customer_msg"]

        if use_cache and item_id in cache.get("main_pipeline", {}):
            cached_res = cache["main_pipeline"][item_id]
            preds["main"]["intent"].append(cached_res["classification"]["intent"])
            preds["main"]["escalate"].append(cached_res["escalation"]["should_escalate"])
            preds["main"]["replies"].append(cached_res["draft_reply"])
        else:
            res = process_message(msg)
            preds["main"]["intent"].append(res["classification"]["intent"])
            preds["main"]["escalate"].append(res["escalation"]["should_escalate"])
            preds["main"]["replies"].append(res["draft_reply"])

            if "main_pipeline" not in cache:
                cache["main_pipeline"] = {}
            cache["main_pipeline"][item_id] = res
            cache_modified = True

        if i % 15 == 0 or i == len(golden_data):
            print(f"      Processed {i}/{len(golden_data)} inquiries...")

    # Step 4: LLM-as-a-Judge Evaluation (on 25 cases)
    judge_sample_size = min(25, len(golden_data))
    if run_judge:
        print(f"\n[4/4] Running LLM-as-a-Judge Rubric Evaluation on {judge_sample_size} cases...")
        if "judge" not in cache:
            cache["judge"] = {}

        for system_key in ["trivial", "simple", "main"]:
            scores = []
            if system_key not in cache["judge"]:
                cache["judge"][system_key] = {}

            for i in range(judge_sample_size):
                item_id = golden_data[i]["id"]
                msg = golden_data[i]["customer_msg"]
                intent = golden_data[i]["true_intent"]
                reply = preds[system_key]["replies"][i]

                if use_cache and item_id in cache["judge"][system_key]:
                    j_res = cache["judge"][system_key][item_id]
                else:
                    j_res = judge.evaluate_reply(msg, intent, reply)
                    cache["judge"][system_key][item_id] = j_res
                    cache_modified = True

                scores.append(j_res)
            preds[system_key]["judge_scores"] = scores

    if cache_modified:
        save_cache(cache)

    elapsed = round(time.time() - start_time, 1)
    print(f"\n[OK] Pipeline evaluations completed in {elapsed} seconds ({elapsed/60:.1f} minutes).")

    # Compute Comparative Metrics
    results = {
        "metadata": {
            "num_evaluated": len(golden_data),
            "execution_time_seconds": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "systems": {}
    }

    for system_key in ["trivial", "simple", "main"]:
        intent_metrics = evaluate_intent_classification(y_true_intent, preds[system_key]["intent"])
        esc_metrics = evaluate_escalation_gate(y_true_escalate, preds[system_key]["escalate"])
        
        # Average judge scores
        j_scores = preds[system_key]["judge_scores"]
        if j_scores:
            avg_rel = round(float(np.mean([s["relevance"] for s in j_scores])), 2)
            avg_grd = round(float(np.mean([s["groundedness"] for s in j_scores])), 2)
            avg_ton = round(float(np.mean([s["tone"] for s in j_scores])), 2)
            avg_act = round(float(np.mean([s["actionability"] for s in j_scores])), 2)
            avg_comp = round(float(np.mean([s["composite_score"] for s in j_scores])), 2)
        else:
            avg_rel = avg_grd = avg_ton = avg_act = avg_comp = 0.0

        results["systems"][system_key] = {
            "intent": {
                "accuracy": intent_metrics["accuracy"],
                "macro_f1": intent_metrics["macro_f1"]
            },
            "escalation": {
                "precision": esc_metrics["precision"],
                "recall": esc_metrics["recall"],
                "f1": esc_metrics["f1"],
                "accuracy": esc_metrics["accuracy"]
            },
            "judge": {
                "relevance": avg_rel,
                "groundedness": avg_grd,
                "tone": avg_ton,
                "actionability": avg_act,
                "composite": avg_comp
            }
        }

    # Print Headline Table
    print("\n" + "=" * 85)
    print(f"{'System':<18} | {'Intent Acc':<10} | {'Intent F1':<10} | {'Esc Recall':<10} | {'Esc F1':<8} | {'Judge Score (1-5)':<15}")
    print("-" * 85)
    for k in ["trivial", "simple", "main"]:
        sys_res = results["systems"][k]
        name = "Trivial Baseline" if k == "trivial" else ("Simple Baseline" if k == "simple" else "Main Pipeline")
        print(f"{name:<18} | {sys_res['intent']['accuracy']*100:>9.1f}% | {sys_res['intent']['macro_f1']:>10.3f} | {sys_res['escalation']['recall']*100:>9.1f}% | {sys_res['escalation']['f1']:>8.3f} | {sys_res['judge']['composite']:>15.2f}")
    print("=" * 85)

    # Save to JSON
    with open(HEADLINE_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save Markdown Summary
    md_content = f"""# Headline Benchmark Results

*Evaluated on {len(golden_data)} stratified golden set examples. Execution time: {elapsed:.1f}s.*

| System | Intent Accuracy | Intent Macro-F1 | Escalation Recall | Escalation F1 | Judge Composite (1-5) | Relevance | Groundedness | Tone | Actionability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline** | {results['systems']['trivial']['intent']['accuracy']*100:.1f}% | {results['systems']['trivial']['intent']['macro_f1']:.3f} | {results['systems']['trivial']['escalation']['recall']*100:.1f}% | {results['systems']['trivial']['escalation']['f1']:.3f} | {results['systems']['trivial']['judge']['composite']:.2f} | {results['systems']['trivial']['judge']['relevance']:.2f} | {results['systems']['trivial']['judge']['groundedness']:.2f} | {results['systems']['trivial']['judge']['tone']:.2f} | {results['systems']['trivial']['judge']['actionability']:.2f} |
| **Simple Baseline**  | {results['systems']['simple']['intent']['accuracy']*100:.1f}% | {results['systems']['simple']['intent']['macro_f1']:.3f} | {results['systems']['simple']['escalation']['recall']*100:.1f}% | {results['systems']['simple']['escalation']['f1']:.3f} | {results['systems']['simple']['judge']['composite']:.2f} | {results['systems']['simple']['judge']['relevance']:.2f} | {results['systems']['simple']['judge']['groundedness']:.2f} | {results['systems']['simple']['judge']['tone']:.2f} | {results['systems']['simple']['judge']['actionability']:.2f} |
| **Main Pipeline**    | **{results['systems']['main']['intent']['accuracy']*100:.1f}%** | **{results['systems']['main']['intent']['macro_f1']:.3f}** | **{results['systems']['main']['escalation']['recall']*100:.1f}%** | **{results['systems']['main']['escalation']['f1']:.3f}** | **{results['systems']['main']['judge']['composite']:.2f}** | **{results['systems']['main']['judge']['relevance']:.2f}** | **{results['systems']['main']['judge']['groundedness']:.2f}** | **{results['systems']['main']['judge']['tone']:.2f}** | **{results['systems']['main']['judge']['actionability']:.2f}** |
"""
    with open(RESULTS_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[OK] Saved results to:\n     - {HEADLINE_RESULTS_JSON.name}\n     - {RESULTS_SUMMARY_MD.name}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run complete evaluation benchmark.")
    parser.add_argument("--limit", type=int, default=24, help="Number of golden set samples (default: 24)")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge scoring for speed")
    parser.add_argument("--no-cache", action="store_true", help="Do not use cached predictions")
    args = parser.parse_args()

    run_benchmark(sample_limit=args.limit, run_judge=not args.no_judge, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
