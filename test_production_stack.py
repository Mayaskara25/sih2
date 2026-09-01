"""
SatQuery AI -- Full Enterprise Production Stack Validation
Person 5: MobileSAM (Component A) + SAHI (Component B) + Innovations 7-10 + TSC.

Verifies under the frozen contract:
  run_vqa(image_path, prompt)     -> {"text_response", "confidence"}
  run_grounding(image_path, target)
      -> {"text_response", "bounding_boxes": [
            {"label", "box_2d": [ymin, xmin, ymax, xmax], "confidence"}], "confidence"}

Tests:
  [1] run_vqa dynamic token softmax confidence
  [2] run_grounding on a large (1600x1200) image -> forces SAHI slicing path
  [3] run_grounding MobileSAM polygon + oriented_box_2d extraction
  [4] contract key presence / shapes
  [5] execution time + VRAM usage + metadata reporting
"""

from __future__ import annotations

import os
import sys
import time
import json

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import numpy as np

import models_registry as reg

CWD = BASE
LARGE_IMG = os.path.join(CWD, "large_test.png")
SMALL_IMG = os.path.join(CWD, "demo_rgb.png")

PASS = 0
FAIL = 0
REPORT = []


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        status = "PASS"
    else:
        FAIL += 1
        status = "FAIL"
    line = f"[{status}] {name}" + (f" -- {detail}" if detail else "")
    print(line)
    REPORT.append({"test": name, "status": status, "detail": detail})


def vram_mb() -> tuple:
    import torch
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        return round(alloc, 1), round(reserved, 1)
    return 0.0, 0.0

def main() -> int:
    import torch

    print("=" * 72)
    print("SatQuery AI -- Production Stack Validation (MobileSAM + SAHI)")
    print("=" * 72)
    print(f"CUDA available: {reg._cuda_available()}")
    print(f"MobileSAM available: {reg._mobilesam_is_available()}")
    print(f"SAHI available (slicing gate): {reg._sahi_is_available()}")
    a, r = vram_mb()
    print(f"VRAM baseline -> allocated {a} MB, reserved {r} MB")

    # ------------------------------------------------------------------ #
    # [1] run_vqa dynamic token softmax confidence
    # ------------------------------------------------------------------ #
    print("\n--- [1] run_vqa (dynamic TSC confidence) ---")
    vqa_img = SMALL_IMG if os.path.isfile(SMALL_IMG) else LARGE_IMG
    t = time.time()
    vqa = reg.run_vqa(vqa_img, "Describe the dominant land cover in this satellite image.")
    dt = time.time() - t
    print(f"  text_response: {vqa['text_response'][:90]!r}")
    print(f"  confidence: {vqa['confidence']:.4f}  (took {dt:.2f}s)")
    check("run_vqa returns text_response (str)", isinstance(vqa["text_response"], str))
    check("run_vqa returns confidence in [0,1]",
          0.0 <= float(vqa["confidence"]) <= 1.0,
          f"conf={vqa['confidence']:.4f}")
    check("VQA confidence is dynamic (not degenerate 1.0)",
          float(vqa["confidence"]) < 0.999)

    # ------------------------------------------------------------------ #
    # [2] run_grounding on large image -> SAHI slicing path
    # ------------------------------------------------------------------ #
    print("\n--- [2] run_grounding large image (SAHI slicing) ---")
    if not os.path.isfile(LARGE_IMG):
        check("large_test.png present", False, "missing large image -- create it")
    else:
        from PIL import Image
        w, h = Image.open(LARGE_IMG).size
        check("large image triggers slicing", w > reg.SAHI_RESIZE_TRIGGER or h > 1024,
              f"{w}x{h} > {reg.SAHI_RESIZE_TRIGGER}")
        t = time.time()
        g = reg.run_grounding(LARGE_IMG, "building")
        dt = time.time() - t
        print(f"  detected boxes: {len(g['bounding_boxes'])} (took {dt:.2f}s)")
        print(f"  text_response: {g['text_response']}")
        check("run_grounding returns text_response (str)",
              isinstance(g["text_response"], str))
        check("run_grounding returns bounding_boxes (list)",
              isinstance(g["bounding_boxes"], list))
        check("run_grounding returns confidence in [0,1]",
              0.0 <= float(g["confidence"]) <= 1.0)

    # ------------------------------------------------------------------ #
    # [3] MobileSAM polygon + oriented_box_2d (Component A)
    # ------------------------------------------------------------------ #
    print("\n--- [3] MobileSAM polygon + OBB extraction ---")
    seg_boxes = []
    for b in g.get("bounding_boxes", []):
        if "oriented_box_2d" in b and "polygon" in b:
            seg_boxes.append(b)
    check("bounding boxes carry oriented_box_2d", len(seg_boxes) > 0,
          f"{len(seg_boxes)}/{len(g.get('bounding_boxes', []))} boxes refined")
    if seg_boxes:
        b0 = seg_boxes[0]
        obb = b0["oriented_box_2d"]
        poly = b0["polygon"]
        check("oriented_box_2d contains angle_deg", "angle_deg" in obb,
              f"theta={obb.get('angle_deg'):.1f}")
        check("oriented_box_2d contains box_2d (contract preserved)",
              "box_2d" in obb and len(obb["box_2d"]) == 4)
        check("polygon is list of [x,y] pairs", isinstance(poly, list) and len(poly) >= 3,
              f"{len(poly)} vertices")

    # ------------------------------------------------------------------ #
    # [4] frozen contract key shapes
    # ------------------------------------------------------------------ #
    print("\n--- [4] frozen contract keys ---")
    req_keys_vqa = {"text_response", "confidence"} <= set(vqa.keys())
    req_keys_g = {"text_response", "bounding_boxes", "confidence"} <= set(g.keys())
    check("run_vqa contract keys present", req_keys_vqa, f"keys={sorted(vqa.keys())}")
    check("run_grounding contract keys present", req_keys_g, f"keys={sorted(g.keys())}")
    if g.get("bounding_boxes"):
        bb = g["bounding_boxes"][0]
        sub = {"label", "box_2d", "confidence"} <= set(bb.keys())
        check("box sub-contract keys present", sub, f"keys={sorted(bb.keys())}")
        check("box_2d has exactly 4 coords [ymin,xmin,ymax,xmax]",
              len(bb["box_2d"]) == 4, f"box_2d={[round(float(v),1) for v in bb['box_2d']]}")

    # ------------------------------------------------------------------ #
    # [5] execution time + VRAM + metadata
    # ------------------------------------------------------------------ #
    print("\n--- [5] runtime / VRAM / metadata ---")
    a, r = vram_mb()
    print(f"  VRAM -> allocated {a} MB, reserved {r} MB")
    if torch.cuda.is_available():
        # "Active" footprint is `allocated`; `reserved` includes allocator-cached
        # free blocks fragmenting during multi-model inference. Budget ~3.0 GB active.
        check("active VRAM under 3.0 GB budget", a <= 3000.0,
              f"allocated={a:.1f} MB (reserved={r:.1f} MB)")
    md = reg.get_execution_metadata()
    print("  metadata:", json.dumps(md, indent=2))
    check("metadata reports MobileSAM", "MobileSAM" in str(md.get("segmentation")))
    check("metadata reports SAHI slicing", "SAHI" in str(md.get("slicing")))

    # ------------------------------------------------------------------ #
    # flush + report
    # ------------------------------------------------------------------ #
    print("\n--- flushing models ---")
    reg.flush_models()
    a2, r2 = vram_mb()
    print(f"  VRAM after flush -> allocated {a2} MB, reserved {r2} MB")

    print("\n" + "=" * 72)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 72)
    out = {"pass": PASS, "fail": FAIL, "report": REPORT,
           "metadata": md, "vram_reserved_mb": r}
    with open(os.path.join(CWD, "production_stack_test_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
