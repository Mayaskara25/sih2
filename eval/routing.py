"""Routing harness: accuracy + per-task confusion matrix over W6's test set.

W8 owns `eval/**` except `eval/routing_testset.json`, which W6 authored and
this harness only SCORES (never edits). It drives the frozen router
(`satquery.controller.router.route`) — not a specialist — because R4/R6's
routing row measures query-text->task selection.

The confusion matrix is built gold-task (rows) x predicted-task (columns), so
off-diagonal cells show exactly which tasks get confused with which. Every
metric carries its sample count (n=...) per honesty rule PLAN.md §5.9.

Pure CPU, no models, no downloads, no GPU — runs in a fraction of a second.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as _c  # noqa: E402  (adds eval/ dir to path; no __init__.py by §5.1)

from satquery.controller.router import route  # noqa: E402

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
TESTSET = os.path.join(_EVAL_DIR, "routing_testset.json")

TASK_ORDER: List[str] = ["vqa", "caption", "grounding", "change", "fusion"]


def load_testset(path: str = TESTSET) -> List[dict]:
    with open(path) as fh:
        return json.load(fh)


def evaluate(queries: List[dict]) -> dict:
    """Run the router over the test set; return metrics + confusion matrix."""
    gold = [q["gold_task"] for q in queries]
    predicted: List[str] = []
    for q in queries:
        decision = route(q["query"], n_inputs=q.get("input_count"))
        predicted.append(decision.task)

    n = len(gold)
    correct = sum(1 for g, p in zip(gold, predicted) if g == p)
    accuracy = correct / n if n else 0.0

    # Confusion matrix: gold (rows) x predicted (cols), in a stable order.
    matrix: Dict[str, Dict[str, int]] = {g: {p: 0 for p in TASK_ORDER} for g in TASK_ORDER}
    wrong: List[dict] = []
    for q, g, p in zip(queries, gold, predicted):
        matrix[g][p] += 1
        if g != p:
            wrong.append({"id": q["id"], "query": q["query"], "gold": g, "predicted": p})

    per_task = {
        t: {
            "correct": int(sum(1 for g, p in zip(gold, predicted) if g == t and p == t)),
            "n": int(sum(1 for g in gold if g == t)),
            "accuracy": (sum(1 for g, p in zip(gold, predicted) if g == t and p == t)
                         / sum(1 for g in gold if g == t)) if gold.count(t) else 0.0,
        }
        for t in TASK_ORDER
    }

    return {
        "accuracy": float(accuracy),
        "n": int(n),
        "correct": int(correct),
        "matrix": matrix,
        "per_task": per_task,
        "wrong": wrong,
    }


def format_confusion_matrix(matrix: Dict[str, Dict[str, int]]) -> str:
    header = ["gold\\pred"] + TASK_ORDER
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for g in TASK_ORDER:
        row = [g] + [str(matrix[g][p]) for p in TASK_ORDER]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(out_json: str | None = None) -> dict:
    queries = load_testset()
    res = evaluate(queries)
    _c_data = {
        "metric": "routing_accuracy",
        "accuracy": res["accuracy"],
        "n": res["n"],
        "date": _c.today(),
        "per_task": res["per_task"],
        "confusion": res["matrix"],
    }
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(_c_data, fh, indent=2)
    print(f"Routing accuracy: {res['correct']}/{res['n']} = {res['accuracy']:.4f}")
    print()
    print(format_confusion_matrix(res["matrix"]))
    if res["wrong"]:
        print("\nMisrouted:")
        for w in res["wrong"]:
            print(f"  {w['id']}: gold={w['gold']} predicted={w['predicted']} :: {w['query']}")
    return _c_data


if __name__ == "__main__":
    main()
