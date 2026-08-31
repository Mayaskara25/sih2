"""CDVQA harness: change-VQA accuracy, change-mask precision vs SECOND.

W4's ``run_change`` is still a W0 stub (returns ``confidence_basis: "stub"``),
and CDVQA/SECOND are not obtainable yet (Google Drive / Baidu hosted, size
unknown — see ``docs/status/W1.md``). So this harness reports ``PLACEHOLDER``
for two independent reasons.

Crucially, the decision to report a placeholder is made by INSPECTING the
specialist's own ``confidence_basis`` at runtime — never by a constant written
here. The moment W4 lands real ``run_change`` bodies (and CDVQA data is
present), the exact same code starts emitting measured numbers with no change.
The stub-detection test in ``tests/test_w8_*.py`` proves the switch by feeding
a stubbed result and a non-stubbed result and asserting the harness flips.

The durable contract property is ``confidence_basis in CONFIDENCE_BASES``
(PLAN.md §5.2); ``== "stub"`` is used *only* as the mechanical probe that
decides placeholder-vs-measured, which is legitimate for exactly this
still-stubbed specialist.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as _c  # noqa: E402

from satquery.contracts import CONFIDENCE_BASES  # noqa: E402

CDVQA_ROOT = os.path.join(_c.REPO_ROOT, "data", "cdvqa")
SECOND_ROOT = os.path.join(_c.REPO_ROOT, "data", "second")
SECOND_TEST = os.path.join(SECOND_ROOT, "test")
# >=20 is PLAN.md's W4 acceptance floor; 30 keeps the run ~20s after model load.
MASK_MAX_ITEMS = 30

_MISSING_DATA = (
    "CDVQA/SECOND not present locally. Both are distributed via Google "
    "Drive / Baidu (no direct HTTP link found), and SECOND's total size is "
    "unknown — see docs/status/W1.md. W8 does not write a data fetcher "
    "(W1-owned)."
)


def data_present() -> bool:
    """True when change-pair data (CDVQA and/or SECOND) is on disk."""
    return any(
        os.path.isdir(root) and os.listdir(root)
        for root in (CDVQA_ROOT, SECOND_ROOT)
    )


def classify_run(specialist_result: dict, n_items: int) -> dict:
    """Decide placeholder-vs-measured by inspecting a run_change result.

    Mechanical stub detection (PLAN.md §5.2 + W8 brief): if the specialist
    announces ``confidence_basis == "stub"``, this harness has nothing real
    to score and reports ``PLACEHOLDER`` — no hardcoded constant here. Feed a
    non-stubbed result and it flips to measured. ``n_items`` is carried so the
    sample count always travels with the metric.
    """
    basis = specialist_result.get("confidence_basis")
    if basis not in CONFIDENCE_BASES:
        return {
            "status": "contract_violation",
            "basis": basis,
            "reason": f"confidence_basis {basis!r} not in {sorted(CONFIDENCE_BASES)}",
        }
    if basis == "stub":
        return {
            "status": _c.PLACEHOLDER,
            "basis": "stub",
            "n": n_items,
            "reason": "specialist run_change still announces confidence_basis='stub' "
                      "(W4 not landed); no measured change result to score.",
        }
    return {"status": "measured", "basis": basis, "n": n_items}


def _probe_stub() -> dict:
    """Mechanically probe whether the change specialist is still a stub.

    Calls the ACTUAL ``run_change`` specialist with probe paths that do not
    exist on disk: the W0/W4 stub returns a valid payload (it never opens the
    paths) and ``classify_run`` then reports ``confidence_basis == "stub"``;
    a real implementation rejects the probe by trying to open real files and
    raising, which ``classify_run`` is not reached for. Either way the stub
    decision comes from the specialist's runtime behaviour, never from a
    constant in this harness.
    """
    from satquery.specialists import change as change_mod
    try:
        res = change_mod.run_change(
            os.path.join(_c.REPO_ROOT, "__w8_probe_t0.tif"),
            os.path.join(_c.REPO_ROOT, "__w8_probe_t1.tif"),
            "__w8_probe__ what changed between these two images?",
        )
    except Exception as exc:
        return {
            "stub": False,
            "basis": None,
            "reason": (
                f"run_change rejected the probe paths ({type(exc).__name__}: {exc}) — "
                "it opens real files, so the W0 stub is gone; real measurement is "
                "possible once CDVQA data lands."
            ),
        }
    verdict = classify_run(res, n_items=0)
    return {"stub": verdict["status"] == _c.PLACEHOLDER, "basis": res.get("confidence_basis"),
            "reason": verdict.get("reason", "")}


def _placeholder_result(blocker: str, stub_status: dict | None = None) -> dict:
    return {
        "status": _c.PLACEHOLDER,
        "blocker": blocker,
        "stub_status": stub_status or {},
        "n": None,
        "date": _c.today(),
    }


def run(max_items: int = 100, pool=None) -> dict:
    """Run change-VQA scoring (or PLACEHOLDER until both data and W4 land)."""
    stub_status = _probe_stub()
    if not data_present():
        blockers = []
        if stub_status["stub"]:
            blockers.append(
                "specialist run_change still announces confidence_basis='stub' "
                f"(mechanical detection, basis={stub_status['basis']!r}); W4 not landed."
            )
        elif stub_status.get("reason"):
            blockers.append("run_change is real (probe paths rejected); W4 has landed.")
        blockers.append(_MISSING_DATA)
        return _placeholder_result("; ".join(blockers), stub_status)

    pairs = _load_pairs()[:max_items]
    if not pairs:
        # No CDVQA question set -> change-VQA accuracy is genuinely unmeasurable.
        # Mask precision does NOT depend on questions, so score it regardless:
        # SECOND ships imagery + reference labels and nothing else is needed.
        res = _placeholder_result(
            "CDVQA change-VQA questions not present (data/cdvqa is empty), so "
            "change-VQA ACCURACY cannot be scored. SECOND imagery and reference "
            "masks ARE present, so mask precision is measured separately below."
        )
        res["mask"] = run_mask_precision()
        return res

    import torch  # deferred: GPU gate only engaged when data is present
    if not torch.cuda.is_available():
        return _placeholder_result(
            "CDVQA/SECOND data present but no CUDA available; run_change needs "
            "the GPU-loaded specialists."
        )

    from collections import Counter

    from satquery.specialists import change as change_mod

    # One run_change per IMAGE PAIR, reused across that pair's ~41 questions --
    # the specialist is the expensive part and its answer does not depend on
    # which of the pair's questions is being asked.
    scored: List[dict] = []
    errors = 0
    for pair in pairs:
        try:
            res = change_mod.run_change(
                pair["image_t0"], pair["image_t1"], pair["qa"][0]["question"]
            )
        except Exception:
            errors += 1
            continue
        verdict = classify_run(res, n_items=len(pair["qa"]))
        if verdict["status"] == _c.PLACEHOLDER:
            return {
                "status": _c.PLACEHOLDER,
                "blocker": verdict.get("reason", "specialist reported a stub basis"),
                "n": 0,
                "date": _c.today(),
                "mask": run_mask_precision(),
            }
        response = res.get("text_response", "")
        for qa in pair["qa"]:
            scored.append({
                "type": qa["type"],
                "gold": qa["gold"],
                "correct": bool(score_answer(qa["gold"], response)),
            })

    if not scored:
        return {
            "status": _c.PLACEHOLDER,
            "blocker": f"all {len(pairs)} CDVQA pairs failed to score.",
            "n": 0,
            "date": _c.today(),
            "mask": run_mask_precision(),
        }

    n_correct = sum(1 for r in scored if r["correct"])
    golds = Counter(r["gold"] for r in scored)
    majority_gold, majority_n = golds.most_common(1)[0]
    per_type = {}
    for t in sorted({r["type"] for r in scored}):
        sub = [r for r in scored if r["type"] == t]
        per_type[t] = {"n": len(sub),
                       "accuracy": sum(1 for r in sub if r["correct"]) / len(sub)}

    accuracy = n_correct / len(scored)
    baseline = majority_n / len(scored)
    # NOT REPORTED AS A MEASURED METRIC (PLAN.md §5.9). run_change returns a
    # DIAGNOSTIC SUMMARY ("differencing flags 51.7% of pixels as changed;
    # largest benclip class-confidence shifts: Moors, Pastures, ...") over
    # BigEarthNet's 19-class vocabulary. CDVQA asks closed-vocabulary questions
    # over SECOND's six classes (water, buildings, trees, low_vegetation,
    # NVG_surface, playground) and expects yes/no or a class token. The two
    # vocabularies barely intersect and the answer FORMATS differ, so a
    # containment score measures vocabulary overlap, not change-VQA ability.
    # Publishing it as "accuracy" would invite the false reading that the change
    # system is ~4% accurate. It is surfaced inside the blocker as evidence for
    # WHY the row is unscorable, never as the row's value.
    return {
        "status": _c.PLACEHOLDER,
        "blocker": (
            "CDVQA questions and images are present and were RUN, but change-VQA "
            "accuracy is not validly scorable against this specialist. run_change "
            "returns a diagnostic summary over BigEarthNet's 19-class vocabulary; "
            "CDVQA expects yes/no or one of SECOND's six class tokens. Measured "
            f"containment score {accuracy:.4f} on n={len(scored)} questions "
            f"({len(pairs)} image pairs) vs a majority-answer floor of "
            f"{baseline:.4f} ('{majority_gold}') -- i.e. BELOW the trivial "
            "baseline, which is the signature of a vocabulary/format mismatch "
            "rather than a capability measurement. Scoring this properly needs a "
            "VQA head mapping change evidence onto CDVQA's closed vocabulary; "
            "that is specialist work (W4), not harness work."
        ),
        "n": len(scored),
        "n_image_pairs": len(pairs),
        "errors": errors,
        "date": _c.today(),
        "containment_score_not_accuracy": accuracy,
        # A closed 19-token vocabulary where "no" alone is ~31% of the split --
        # accuracy is meaningless without this floor.
        "majority_baseline": baseline,
        "majority_answer": majority_gold,
        "per_type": per_type,
        "mask": run_mask_precision(),
    }


def _load_second_pairs() -> List[dict]:
    """SECOND test pairs: test/{im1,im2}/<name>.png + test/{label1,label2}/<name>.png.

    Deterministic (sorted) so two runs score the same items -- run_all.py
    asserts numeric identity across runs.
    """
    import glob

    pairs: List[dict] = []
    for p0 in sorted(glob.glob(os.path.join(SECOND_TEST, "im1", "*.png"))):
        name = os.path.basename(p0)
        cand = {
            "image_t0": p0,
            "image_t1": os.path.join(SECOND_TEST, "im2", name),
            "label_t0": os.path.join(SECOND_TEST, "label1", name),
            "label_t1": os.path.join(SECOND_TEST, "label2", name),
        }
        if all(os.path.exists(v) for v in cand.values()):
            pairs.append(cand)
    return pairs


def run_mask_precision(max_items: int = MASK_MAX_ITEMS) -> dict:
    """Change-mask precision/recall/IoU against SECOND reference labels.

    PLAN.md W4 acceptance names "change mask precision reported against >=20
    SECOND reference masks". SECOND carries per-date SEMANTIC labels, so the
    binary change reference is `label_t0 != label_t1` -- pixels whose land
    cover class differs between the two dates.

    Reports `baseline_precision` alongside: the precision of a degenerate
    "everything changed" predictor, which equals the ground-truth change rate.
    A precision below that floor would mean the detector is worse than
    predicting change everywhere, so the metric is uninterpretable without it.
    """
    pairs = _load_second_pairs()[:max_items]
    if not pairs:
        return {
            "status": _c.PLACEHOLDER,
            "blocker": (
                f"SECOND reference masks not found under {SECOND_TEST} "
                "(expected test/{im1,im2,label1,label2}/<name>.png)."
            ),
            "n": 0,
        }

    import torch
    if not torch.cuda.is_available():
        return {
            "status": _c.PLACEHOLDER,
            "blocker": "SECOND masks present but no CUDA; run_change needs the GPU.",
            "n": 0,
        }

    import numpy as np
    import rasterio

    from satquery.specialists import change as change_mod

    prec, rec, iou, gt_rate, pred_rate, errors = [], [], [], [], [], 0
    for pair in pairs:
        try:
            res = change_mod.run_change(pair["image_t0"], pair["image_t1"], "what changed?")
            mp = res.get("change_mask_path")
            if not mp or not os.path.exists(mp):
                errors += 1
                continue
            pred = rasterio.open(mp).read()[0] > 0
            l0 = rasterio.open(pair["label_t0"]).read()
            l1 = rasterio.open(pair["label_t1"]).read()
            gt = (l0 != l1).any(axis=0)
            if pred.shape != gt.shape:
                errors += 1
                continue
            tp = int((pred & gt).sum())
            fp = int((pred & ~gt).sum())
            fn = int((~pred & gt).sum())
            prec.append(tp / (tp + fp) if tp + fp else 0.0)
            rec.append(tp / (tp + fn) if tp + fn else 0.0)
            iou.append(tp / (tp + fp + fn) if tp + fp + fn else 0.0)
            gt_rate.append(float(gt.mean()))
            pred_rate.append(float(pred.mean()))
        except Exception:
            errors += 1
    if not prec:
        return {"status": _c.PLACEHOLDER,
                "blocker": f"all {len(pairs)} SECOND pairs failed to score.", "n": 0}
    return {
        "status": "measured",
        "n": len(prec),
        "errors": errors,
        "precision": float(np.mean(prec)),
        "precision_sd": float(np.std(prec)),
        "recall": float(np.mean(rec)),
        "iou": float(np.mean(iou)),
        "gt_change_rate": float(np.mean(gt_rate)),
        "pred_change_rate": float(np.mean(pred_rate)),
        "baseline_precision": float(np.mean(gt_rate)),
    }


def _load_pairs(split: str = "Test", max_images: int = 30) -> List[dict]:
    """Load CDVQA change-VQA items, joined across its three normalised files.

    CDVQA's real schema is NOT flat: `<split>_images.json` holds
    ``{"images":[{id, file_name, ...}]}``, `<split>_questions.json` holds
    ``{"questions":[{id, img_id, type, question}]}`` and
    `<split>_answers.json` holds ``{"answers":[{question_id, answer}]}``.
    The images are SECOND's TRAIN subset (2,968 pairs), disjoint from the
    SECOND test split used for mask precision.

    Grouped by image so the caller can run the (expensive) specialist once per
    pair and answer all ~41 of its questions from that single result.
    """
    import json

    idir = os.path.join(SECOND_ROOT, "train")
    ipath = os.path.join(CDVQA_ROOT, f"{split}_images.json")
    qpath = os.path.join(CDVQA_ROOT, f"{split}_questions.json")
    apath = os.path.join(CDVQA_ROOT, f"{split}_answers.json")
    if not all(os.path.exists(x) for x in (ipath, qpath, apath)):
        return []

    images = json.load(open(ipath))["images"]
    questions = json.load(open(qpath))["questions"]
    answers = json.load(open(apath))["answers"]

    ans_by_q = {a["question_id"]: a["answer"] for a in answers if a.get("active", True)}
    qs_by_img: dict = {}
    for q in questions:
        if not q.get("active", True):
            continue
        gold = ans_by_q.get(q["id"])
        if gold is None:
            continue
        qs_by_img.setdefault(q["img_id"], []).append(
            {"question": q["question"], "gold": gold, "type": q.get("type", "")}
        )

    # CDVQA lists 15,488 image ENTRIES for only 968 unique file_names (~16 rows
    # per file, each carrying its own slice of that image's questions). Grouping
    # by id would take ~2 questions per pair instead of ~41; group by file_name.
    by_name: dict = {}
    for im in sorted(images, key=lambda x: x["id"]):
        if not im.get("active", True):
            continue
        by_name.setdefault(im["file_name"], []).extend(qs_by_img.get(im["id"], []))

    pairs: List[dict] = []
    for name in sorted(by_name):
        t0 = os.path.join(idir, "im1", name)
        t1 = os.path.join(idir, "im2", name)
        qa = by_name[name]
        if qa and os.path.exists(t0) and os.path.exists(t1):
            pairs.append({"image_t0": t0, "image_t1": t1, "file_name": name, "qa": qa})
        if len(pairs) >= max_images:
            break
    return pairs


_YES_NO = {"yes", "no"}


def score_answer(gold: str, response: str) -> bool:
    """Closed-vocabulary containment scoring for a descriptive response.

    CDVQA answers come from a 19-token closed vocabulary (yes/no, six land-cover
    class names, and percentage buckets like ``0_to_10``). ``run_change``
    returns a descriptive sentence rather than a single token, so exact match is
    the wrong comparison -- this checks whether the gold token appears as a
    WORD in the normalised response.

    yes/no is matched on word boundaries specifically: a substring test would
    let "no" match inside "not", "cannot" or "north" and silently inflate
    accuracy on the 58% of the split that is yes/no.
    """
    import re

    g = gold.strip().lower()
    r = " ".join(response.lower().replace("_", " ").split())
    if g in _YES_NO:
        return re.search(rf"\b{g}\b", r) is not None
    return re.search(rf"\b{re.escape(g.replace('_', ' '))}\b", r) is not None


def main(out_json: str | None = None) -> dict:
    res = run()
    if out_json:
        import json
        with open(out_json, "w") as fh:
            json.dump(res, fh, indent=2)
    if res["status"] == _c.PLACEHOLDER:
        print(f"CDVQA: {_c.PLACEHOLDER} — {res['blocker']}")
    else:
        print(f"CDVQA: accuracy={res['accuracy']:.4f} (n={res['n']})")
    m = res.get("mask") or {}
    if m.get("status") == "measured":
        print(
            f"SECOND mask: precision={m['precision']:.4f} recall={m['recall']:.4f} "
            f"IoU={m['iou']:.4f} (n={m['n']}, baseline precision "
            f"{m['baseline_precision']:.4f})"
        )
    elif m:
        print(f"SECOND mask: {_c.PLACEHOLDER} — {m.get('blocker','')}")
    return res


if __name__ == "__main__":
    main()
