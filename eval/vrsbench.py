"""VRSBench harness: BLEU/ROUGE/CIDEr on captioning, IoU on grounding.

VRSBench is NOT downloaded (PLAN.md §2.2a / W1 status: it is the largest
local item, ~4 GB eval images, and W1's fetcher has not landed it — see
``docs/status/W1.md``). Until the prescribed test split is present, every
metric reports ``PLACEHOLDER`` with the blocker named, per PLAN.md §5.9 —
a placeholder always beats a guessed number.

When the data lands (under ``data/vrsbench/`` per W1's manifest), the
``run()`` path becomes live without a harness rewrite: it iterates the test
split, produces captions through ``run_query`` (W3's ``run_caption``) and
boxes through ``run_grounding``, and computes BLEU/ROUGE-L/CIDEr-proxy (see
``_common.cidre`` — an explicit proxy, not official MS-COCO CIDEr) plus
box IoU against the reference. Grounding requires GPU gating because W3's
specialists load real weights.

No ``eval/__init__.py`` (PLAN.md §5.1), so sibling imports are flat.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as _c  # noqa: E402

VRSBENCH_ROOT = os.path.join(_c.REPO_ROOT, "data", "vrsbench")
_MISSING_DATA = (
    f"data/vrsbench/ eval split not present. "
    f"VRSBench is not downloaded (PLAN.md §2.2a / W1): the ~4 GB validation "
    f"image set is the largest local item and its fetcher has not landed. "
    f"W8 does not write a data fetcher (W1-owned)."
)


def data_present() -> bool:
    """True when the VRSBench test split is on disk."""
    # W1's fetcher will write a manifest here; until then, presence is judged
    # by the expected eval-directory marker.
    marker = os.path.join(VRSBENCH_ROOT, "eval")
    return os.path.isdir(marker) and any(
        os.path.isfile(os.path.join(marker, f))
        for f in os.listdir(marker) if not f.startswith(".")
    )


def _placeholder_result(blocker: str) -> dict:
    return {
        "status": _c.PLACEHOLDER,
        "blocker": blocker,
        "n": None,
        "date": _c.today(),
        "bleu": None,
        "rouge_l": None,
        "cider_proxy": None,
        "grounding_iou": None,
    }


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """IoU over [ymin, xmin, ymax, xmax] boxes (contract order)."""
    ay1, ax1, ay2, ax2 = box_a
    by1, bx1, by2, bx2 = box_b
    iy1, ix1 = max(ay1, by1), max(ax1, bx1)
    iy2, ix2 = min(ay2, by2), min(ax2, bx2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run(max_items: int = 100) -> dict:
    """Return VRSBench metrics (or PLACEHOLDER while the data is absent).

    The real path is written and unit-testable independently of data; the
    GPU dispatch is gated on data presence at the call site (::func:`run`) so
    a GPU-less or data-less clone never triggers a download.
    """
    if not data_present():
        return _placeholder_result(_MISSING_DATA)

    # --- LIVE path (fires only once data/vrsbench/eval exists) -------------
    import torch  # deferred so importing this module never requires torch/cuda
    if not torch.cuda.is_available():
        return _placeholder_result(
            "data/vrsbench/ eval split present but no CUDA available; "
            "grounding/captioning need the GPU-loaded specialists."
        )

    from satquery.controller.trace import run_query

    items = _load_eval_items()[:max_items]
    refs: List[str] = [it["caption"] for it in items]
    cands: List[str] = []
    ious: List[float] = []
    n_iou = 0
    for it in items:
        trace = run_query("Describe this image.", [it["image_path"]])
        cands.append(trace["result"].get("text_response", ""))
        gtrace = run_query(f"Where is the {it['target']}?", [it["image_path"]])
        boxes = gtrace["result"].get("bounding_boxes", [])
        if boxes and it["reference_box"]:
            ious.append(_iou(boxes[0]["box_2d"], it["reference_box"]))
            n_iou += 1

    return {
        "status": "measured",
        "n": len(cands),
        "n_iou": n_iou,
        "date": _c.today(),
        "bleu": _c.bleu_corpus(refs, cands),
        "rouge_l": _c.rouge_l_corpus(refs, cands),
        "cider_proxy": _c.cidre(refs, cands),
        "grounding_iou": (sum(ious) / len(ious)) if ious else 0.0,
    }


def _load_eval_items() -> List[dict]:
    """Load VRSBench eval items (JSONL under data/vrsbench/eval/)."""
    import glob
    import json

    files = sorted(glob.glob(os.path.join(VRSBENCH_ROOT, "eval", "*.jsonl")))
    items: List[dict] = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    items.append(json.loads(line))
    return items


def main(out_json: str | None = None) -> dict:
    res = run()
    if out_json:
        import json
        with open(out_json, "w") as fh:
            json.dump(res, fh, indent=2)
    if res["status"] == _c.PLACEHOLDER:
        print(f"VRSBench: {_c.PLACEHOLDER} — {res['blocker']}")
    else:
        print(f"VRSBench: BLEU={res['bleu']:.4f} ROUGE-L={res['rouge_l']:.4f} "
              f"CIDEr={res['cider_proxy']:.4f} IoU={res['grounding_iou']:.4f} (n={res['n']})")
    return res


if __name__ == "__main__":
    main()
