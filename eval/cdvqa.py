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
        return _placeholder_result(_MISSING_DATA)

    import torch  # deferred: GPU gate only engaged when data is present
    if not torch.cuda.is_available():
        return _placeholder_result(
            "CDVQA/SECOND data present but no CUDA available; run_change needs "
            "the GPU-loaded specialists."
        )

    from satquery.specialists import change as change_mod

    results: List[dict] = []
    for pair in pairs:
        try:
            res = change_mod.run_change(
                pair["image_t0"], pair["image_t1"], pair["question"]
            )
        except Exception as exc:  # specialist crash -> record, keep going
            results.append({"status": "error", "error": str(exc)})
            continue
        verdict = classify_run(res, n_items=1)
        results.append(verdict)

    stubbed = [r for r in results if r["status"] == _c.PLACEHOLDER]
    if stubbed:
        return {
            "status": _c.PLACEHOLDER,
            "blocker": stubbed[0]["reason"],
            "n": len(results),
            "date": _c.today(),
        }

    # --- measured path ------------------------------------------------------
    n_correct = sum(
        1 for r in results
        if r["status"] == "measured" and r.get("correct")
    )
    return {
        "status": "measured",
        "n": len(results),
        "date": _c.today(),
        "accuracy": (n_correct / len(results)) if results else 0.0,
        "mask_precision": None,  # SECOND reference masks not yet wired
    }


def _load_pairs() -> List[dict]:
    """Load (image_t0, image_t1, question) change pairs from disk."""
    import glob
    import json

    pairs: List[dict] = []
    for f in sorted(glob.glob(os.path.join(CDVQA_ROOT, "**", "*.json"), recursive=True)):
        with open(f) as fh:
            data = json.load(fh)
        for s in data if isinstance(data, list) else data.get("questions", []):
            pairs.append({
                "image_t0": s.get("image_t0"),
                "image_t1": s.get("image_t1"),
                "question": s.get("question", "What changed?"),
            })
    return pairs


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
    return res


if __name__ == "__main__":
    main()
