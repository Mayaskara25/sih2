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

    items = _load_eval_items(max_items)[:max_items]
    if not items:
        # The eval JSONs can be on disk while the 3.8 GB image zip is not yet
        # extracted; scoring an empty list would report 0.0 as if measured.
        return _placeholder_result(
            "VRSBench eval annotations are present but no validation IMAGES were "
            f"found under {VRSBENCH_ROOT} (expected an extracted Images_val/ "
            "directory). Scoring an empty item list would publish 0.0 as if it "
            "were measured, so this reports PLACEHOLDER instead."
        )
    refs: List[str] = [it["caption"] for it in items]
    cands: List[str] = []
    ious: List[float] = []
    n_iou = 0

    # TWO SEQUENTIAL PASSES, NOT ONE INTERLEAVED LOOP. PLAN.md §4.3 keeps a
    # single heavy model resident: "acquiring a different heavy role auto-
    # releases the currently resident one". Captioning uses the VLM and
    # grounding uses Grounding DINO, so alternating them per item evicts and
    # reloads a multi-GB model on EVERY call -- ~2N loads for N items, each
    # ~20s of disk/CPU work with the GPU idle (measured: utilisation 0-5%,
    # VRAM sawtoothing 761 <-> 2100 MiB). Batching by task makes it 2 loads
    # total. Identical results; the only thing that changes is model residency.
    for it in items:
        trace = run_query("Describe this image.", [it["image_path"]])
        cands.append((trace.get("result") or {}).get("text_response", ""))

    for it in items:
        if not it["reference_box"]:
            continue
        gtrace = run_query(f"Where is the {it['target']}?", [it["image_path"]])
        boxes = (gtrace.get("result") or {}).get("bounding_boxes", [])
        if boxes:
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


IMAGE_DIRS = ("Images_val", "images_val", "Images", "images", "val")


def _find_image_dir() -> str | None:
    """Locate the extracted validation image directory, whatever it unzipped to."""
    for d in IMAGE_DIRS:
        p = os.path.join(VRSBENCH_ROOT, d)
        if os.path.isdir(p) and os.listdir(p):
            return p
    # fall back: any directory under the root holding .png files
    if os.path.isdir(VRSBENCH_ROOT):
        for name in sorted(os.listdir(VRSBENCH_ROOT)):
            p = os.path.join(VRSBENCH_ROOT, name)
            if os.path.isdir(p) and any(f.endswith(".png") for f in os.listdir(p)[:50]):
                return p
    return None


def _parse_ref_box(gt: str, width: int, height: int) -> List[float] | None:
    """VRSBench referring box -> the contract's PIXEL [ymin, xmin, ymax, xmax].

    TWO conversions are required and BOTH matter; skipping either silently
    produces a meaningless IoU rather than an error:

      * scale -- VRSBench encodes ``{<x1><y1><x2><y2>}`` on a 0-100 grid
        (verified: coordinate range across the split is exactly 0-100, and the
        ordering was cross-checked against the dataset's own ``obj_corner``
        polygon field). ``run_grounding`` returns PIXELS.
      * axis order -- VRSBench is (x, y, x, y); the box_2d contract and
        ``_iou`` are (y, x, y, x).
    """
    import re

    v = [int(x) for x in re.findall(r"<(\d+)>", gt or "")]
    if len(v) != 4:
        return None
    x1, y1, x2, y2 = v
    return [y1 / 100.0 * height, x1 / 100.0 * width,
            y2 / 100.0 * height, x2 / 100.0 * width]


def _load_eval_items(max_items: int = 100) -> List[dict]:
    """Join VRSBench's caption and referring eval files into scorable items.

    VRSBench ships three flat JSON arrays keyed by ``image_id``, not the JSONL
    this harness originally assumed. Captioning supplies the reference caption;
    referring supplies the grounding target phrase and its box. Only images
    present in BOTH (9,318 of 9,350) yield a fully scorable item.
    """
    import json

    img_dir = _find_image_dir()
    if img_dir is None:
        return []

    cap_path = os.path.join(VRSBENCH_ROOT, "eval", "VRSBench_EVAL_Cap.json")
    ref_path = os.path.join(VRSBENCH_ROOT, "eval", "VRSBench_EVAL_referring.json")
    if not (os.path.exists(cap_path) and os.path.exists(ref_path)):
        return []

    caps = json.load(open(cap_path))
    refs = json.load(open(ref_path))
    ref_by_img: dict = {}
    for r in refs:
        ref_by_img.setdefault(r["image_id"], r)  # first referring expr per image

    from PIL import Image

    items: List[dict] = []
    for c in sorted(caps, key=lambda x: str(x.get("image_id"))):
        img_id = c["image_id"]
        path = os.path.join(img_dir, img_id)
        if not os.path.exists(path):
            continue
        r = ref_by_img.get(img_id)
        box = None
        target = None
        if r is not None:
            try:
                with Image.open(path) as im:
                    w, h = im.size
                box = _parse_ref_box(r.get("ground_truth", ""), w, h)
                target = r.get("question")
            except Exception:
                box = None
        items.append({
            "image_path": path,
            "caption": c.get("ground_truth", ""),
            "target": target or "the main object",
            "reference_box": box,
        })
        if len(items) >= max_items:
            break
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
