"""RSVQA-LR VQA harness — the W8 priority: real data + real specialist.

Drives the full agentic round trip (``satquery.controller.trace.run_query``,
per PLAN.md §5.3/W8 brief: measure through the controller, not by importing
the specialist directly) over the RSVQA-LR **test** split and reports:

  * raw exact-match accuracy
  * case-normalised exact-match accuracy (the trap the W8 brief warns about:
    ``run_vqa`` returns both ``Yes`` and ``yes``, so we lowercase+strip both
    sides before comparing and report raw vs normalised SIDE BY SIDE so the
    effect is visible, never hidden)
  * answer-match rate (gold answer token present in the normalised prediction)
  * BLEU (with a note that RSVQA gold answers are single words/numbers, so the
    useful signal is BLEU-1; higher-n collapses to the single token)

Because routing is query-text-driven (R4) and several RSVQA queries genuinely
misroute away from ``vqa`` (e.g. "Is it a rural or an urban *area*" -> grounding
via exemplar-NN — see docs/status/W8.md), the harness PRE-ROUTES with the CPU
router and only dispatches the GPU ``run_query`` for queries routed to ``vqa``.
The questions the router sends elsewhere are counted and reported as a
routing-fidelity finding; we never force a route or import the specialist
directly, and we never fabricate a VQA answer for a query the router routed to
grounding.

Deterministic subsampling (fixed seed) makes two consecutive runs identical.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as _c  # noqa: E402

_DATA_DIR = os.path.join(_c.REPO_ROOT, "data", "rsvqa_lr")
_IMAGES_DIR = os.path.join(_DATA_DIR, "Images_LR")


def _load_split(split: str) -> dict:
    """Return {id: record} for ``questions``/``answers``/``images`` active items."""
    with open(os.path.join(_DATA_DIR, f"LR_split_test_{split}.json")) as fh:
        data = json.load(fh)
    key = {"questions": "questions", "answers": "answers", "images": "images"}[split]
    return {x["id"]: x for x in data[key] if x.get("active")}


def _image_path_for(img_id: int) -> str:
    return os.path.join(_IMAGES_DIR, f"{img_id}.tif")


def load_test_set() -> List[dict]:
    """Build [{id, type, question, gold_answer, image_path}] for active test QA."""
    questions = _load_split("questions")
    answers = _load_split("answers")
    images = _load_split("images")
    out: List[dict] = []
    for qid, q in questions.items():
        ans = answers.get(qid, {}).get("answer")
        img = images.get(q["img_id"])
        path = _image_path_for(q["img_id"]) if img and os.path.isfile(_image_path_for(q["img_id"])) else None
        out.append({
            "id": qid,
            "type": q["type"],
            "question": q["question"],
            "gold_answer": ans,
            "image_path": path,
            "routed_task": None,
            "prediction": None,
        })
    return out


# Uncertainty when a permutation of the answer vocabulary is present.
_ANSWER_VOCAB = {
    "presence": {"yes", "no"},
    "comp": {"yes", "no"},
    "rural_urban": {"rural", "urban"},
    "count": None,  # numeric
}


def ground_answer_tokens(gold: str, qtype: str) -> List[str]:
    """Normalised gold tokens that must appear in a correct prediction.

    For multi-word golds (e.g. a count reported as a number) the whole
    normalised gold string is the required token set. For yes/no and
    rural/urban the vocabulary is the single distinguishing word, so a
    correctly-answered prediction contains exactly that word.
    """
    vocab = _ANSWER_VOCAB.get(qtype)
    if vocab:
        norm = _c.strip_norm(gold)
        if norm in vocab:
            return [norm]
    return _c.tokenise(gold)


def score_pair(prediction: str, gold: str, qtype: str) -> Dict[str, bool or float]:
    """Per-question scores for one (prediction, gold) pair."""
    raw_match = (prediction == gold)
    pred_norm = _c.strip_norm(prediction)
    gold_norm = _c.strip_norm(gold)
    norm_match = (pred_norm == gold_norm)
    required = ground_answer_tokens(gold, qtype)
    pred_tokens = _c.tokenise(prediction)
    answer_match = bool(required) and all(t in pred_tokens for t in required)
    return {
        "raw_match": bool(raw_match),
        "norm_match": bool(norm_match),
        "answer_match": bool(answer_match),
        "bleu1": float(_c.bleu1(gold, prediction)),
    }


def evaluate(items: List[dict]) -> dict:
    """Aggregate per-question scores into the harness metrics."""
    scored = [it for it in items if it["routed_task"] == "vqa" and it["prediction"] is not None]
    n = len(scored)
    raw_acc = sum(1 for it in scored if it["scores"]["raw_match"]) / n if n else 0.0
    norm_acc = sum(1 for it in scored if it["scores"]["norm_match"]) / n if n else 0.0
    match_rate = sum(1 for it in scored if it["scores"]["answer_match"]) / n if n else 0.0
    bleu1 = sum(it["scores"]["bleu1"] for it in scored) / n if n else 0.0

    misrouted = [it for it in items if it["routed_task"] != "vqa"]
    routed_fidelity = (len(items) - len(misrouted)) / len(items) if items else 0.0

    per_type: Dict[str, dict] = {}
    types = sorted({it["type"] for it in scored})
    for t in types:
        sub = [it for it in scored if it["type"] == t]
        per_type[t] = {
            "n": len(sub),
            "accuracy_normalised": sum(1 for it in sub if it["scores"]["norm_match"]) / len(sub),
        }

    return {
        "n_scored_vqa": n,
        "n_misrouted": len(misrouted),
        "n_total": len(items),
        "routing_fidelity": routed_fidelity,
        "raw_accuracy": raw_acc,
        "normalised_accuracy": norm_acc,
        "answer_match_rate": match_rate,
        "bleu1": bleu1,
        "normalisation": "lowercase + strip + collapse whitespace on both sides",
        "per_type": per_type,
    }


def run(sample_size: int = 160, seed: int = _c.DEFAULT_SEED) -> dict:
    """Run the RSVQA-LR VQA evaluation through the controller.

    ``sample_size`` bounds the deterministic subset that is routed; only the
    vqa-routed subset is dispatched on GPU. Returns the metrics dict.
    """
    full = load_test_set()
    if not full:
        return {"status": _c.PLACEHOLDER,
                "blocker": "data/rsvqa_lr test split not present on disk.",
                "n_scored_vqa": 0, "n_misrouted": 0, "n_total": 0}
    subset = _c.seeded_subset(full, sample_size, seed=seed)

    import torch
    if not torch.cuda.is_available():
        return {"status": _c.PLACEHOLDER,
                "blocker": "RSVQA data present but no CUDA: run_vqa needs the GPU.",
                "n_scored_vqa": 0, "n_misrouted": 0, "n_total": len(subset)}

    from satquery.controller.router import route
    from satquery.controller.trace import run_query

    for it in subset:
        decision = route(it["question"], n_inputs=1)
        it["routed_task"] = decision.task
        if it["routed_task"] != "vqa" or not it["image_path"]:
            it["prediction"] = None
            continue
        trace = run_query(it["question"], [it["image_path"]])
        it["prediction"] = trace["result"].get("text_response", "")
        it["scores"] = score_pair(it["prediction"], it["gold_answer"], it["type"])

    res = evaluate(subset)
    res["status"] = "measured"
    res["date"] = _c.today()
    res["sample_size"] = len(subset)
    res["seed"] = seed
    return res


def format_report(res: dict) -> str:
    lines = [
        "RSVQA-LR VQA (through controller run_query):",
        f"  sample routed: {res.get('n_total', 0)}  (seed {res.get('seed')})",
        f"  routed to vqa and scored: {res['n_scored_vqa']}",
        f"  misrouted by router (routing fidelity {res.get('routing_fidelity', 0.0):.3f}): {res['n_misrouted']}",
        f"  raw exact-match accuracy:     {res['raw_accuracy']:.4f}",
        f"  normalised accuracy:          {res['normalised_accuracy']:.4f}",
        f"  answer-match rate:            {res['answer_match_rate']:.4f}",
        f"  BLEU-1:                       {res['bleu1']:.4f}",
        f"  normalisation:                {res['normalisation']}",
    ]
    return "\n".join(lines)


def main(out_json: str | None = None, sample_size: int = 160) -> dict:
    res = run(sample_size=sample_size)
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(res, fh, indent=2)
    if res.get("status") in (None, "measured") and "raw_accuracy" in res:
        print(format_report(res))
    else:
        print(f"RSVQA: {res.get('status', _c.PLACEHOLDER)} — {res.get('blocker', '')}")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="RSVQA-LR VQA harness")
    ap.add_argument("--sample-size", type=int, default=160)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(out_json=args.out, sample_size=args.sample_size)
