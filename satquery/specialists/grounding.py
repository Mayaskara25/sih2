"""
Text-guided visual grounding specialist.

STATUS: REAL (W3). Grounding DINO tiny via satquery.runtime.modelpool.

Contract: PLAN.md §4.1 / satquery.contracts.{GroundingResult,BoundingBox}

Implementation notes (measured on this machine, transformers 5.16.1 / torch
2.13.0+cu130 / GTX 1650):

  - box_2d is returned as [ymin, xmin, ymax, xmax] in PIXEL coordinates of
    `raster.display_rgb` (the same array handed to the model). Grounding DINO
    natively returns [xmin, ymin, xmax, ymax] -- converted below.
  - `post_process_grounded_object_detection` takes `threshold=` in 5.x, not
    the 4.x `box_threshold=`. No version-straddling try/except (PLAN.md
    §2.2a / W0 note).
  - **fp16 for grounding-dino-tiny crashes on this stack**: the deformable
    attention's `grid_sample` call mixes a float32 sampling grid with half
    feature maps ("expected scalar type Half but found Float"), reproduced
    directly against `IDEA-Research/grounding-dino-tiny` outside this
    module. PLAN.md §3.2 calls for fp16, but W0's default RoleSpec loader
    (satquery/runtime/modelpool.py::_load_grounding) is frozen and hardcodes
    fp16 for CUDA. Rather than edit that frozen file, this module registers
    a corrected fp32 RoleSpec for "grounding" via the documented
    `ModelPool.register()` extension point (see `_ensure_fp32_grounding_role`
    below) before acquiring it. fp32 grounding-dino-tiny measures ~1.55 GB
    peak allocated (see docs/status/W3.md) -- well inside the ~2.5 GB budget
    with W2's benclip/CLIP not resident. This divergence is disclosed in
    docs/status/W3.md as contract pressure on PLAN.md §3.2's precision table.
  - Grounding DINO was trained on lowercase phrases ending in a period
    ("a road."); `_normalize_query` enforces that.
  - Label coercion: transformers 5.x's `post_process_grounded_object_detection`
    returns clean `text_labels` in the common case, but different transformer
    versions have shipped labels under `labels` instead, as bare ints, or as
    empty strings -- `_coerce_label` resolves all of that and always falls
    back to the (stripped, un-normalized) query text rather than emitting a
    stray int or empty string into the contract payload.
  - Zero detections is a legitimate, honestly-reported outcome: empty
    `bounding_boxes`, a clear `text_response`, `confidence=0.0`.
  - `overlay_path` is always populated with a real rendered image (boxes
    drawn, or a plain copy of the image when there are zero detections) so
    R7's "grounding is visibly reachable" holds even in the empty case.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from satquery.contracts import GroundingResult, validate_grounding_result
from satquery.io.raster import load_raster
from satquery.runtime.modelpool import DEFAULT_REGISTRY, RoleSpec, model_pool

# Where rendered overlays are written. `runs/` is gitignored (PLAN.md §5.7)
# and is already the convention for run artifacts (PLAN.md §4.2 `artifacts`).
_OVERLAY_DIR = os.path.join("runs", "artifacts", "grounding")

_DETECTION_THRESHOLD = float(os.environ.get("SATQUERY_GROUND_THRESHOLD", "0.20"))
_TEXT_THRESHOLD = float(os.environ.get("SATQUERY_GROUND_TEXT_THRESHOLD", "0.20"))

_fp32_role_registered = False


def _load_grounding_fp32(spec: RoleSpec) -> Tuple[Any, Any]:
    """Corrected loader for the "grounding" role: fp32 on CUDA and CPU alike.

    See the module docstring for why fp16 (W0's default loader) crashes on
    this transformers/torch combination. Mirrors
    satquery.runtime.modelpool._load_grounding otherwise (AutoModel /
    AutoProcessor, .eval(), device placement).
    """
    import torch as _torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        spec.model_id, dtype=_torch.float32, trust_remote_code=True
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(spec.model_id, trust_remote_code=True)
    return model, processor


def _ensure_fp32_grounding_role() -> None:
    """Register the fp32 grounding RoleSpec once, overriding W0's fp16
    default via the documented `ModelPool.register()` extension point
    (modelpool.py docstring: "Add or override a role spec. Used by tests to
    inject fake loaders." -- this is the same mechanism, used for a real
    correctness fix rather than a test double). Idempotent."""
    global _fp32_role_registered
    if _fp32_role_registered:
        return
    default_spec = DEFAULT_REGISTRY.get("grounding")
    model_id = default_spec.model_id if default_spec is not None else os.environ.get(
        "SATQUERY_GROUNDING_MODEL", "IDEA-Research/grounding-dino-tiny"
    )
    model_pool.register(
        RoleSpec(
            role="grounding",
            model_id=model_id,
            precision="fp32",  # accurate precision string -> honest trace (§4.5)
            loader=_load_grounding_fp32,
            exempt=False,
        )
    )
    _fp32_role_registered = True


def _normalize_query(target_text: str) -> str:
    """Grounding DINO expects lowercase phrase(s) ending in a period."""
    q = (target_text or "").strip().lower()
    if not q:
        q = "object"
    if not q.endswith("."):
        q = f"{q}."
    return q


def _coerce_label(raw: Any, fallback: str) -> str:
    """Resolve a single detection's label robustly across transformers
    versions: `text_labels` entries are normally clean strings, but some
    versions have returned bare class ids or empty strings under `labels`.
    Always fall back to the (human) query text rather than emit a stray
    int/empty string into the contract payload."""
    if raw is None:
        return fallback
    text = str(raw).strip()
    if not text or text.isdigit():
        return fallback
    return text


def _to_contract_box(xyxy: Any, raw_label: Any, raw_score: Any, fallback_label: str) -> Dict[str, Any]:
    """Convert one native Grounding DINO detection to a §4.1 BoundingBox dict.

    `xyxy` is [xmin, ymin, xmax, ymax] (Grounding DINO's native order); the
    contract order is [ymin, xmin, ymax, xmax] (PLAN.md §4.1). Also clamps
    ymin<=ymax / xmin<=xmax (the validator rejects the inverted case) and
    coerces every number with float() (PLAN.md §4.5).
    """
    xmin, ymin, xmax, ymax = [float(v) for v in xyxy]
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    return {
        "label": _coerce_label(raw_label, fallback_label),
        "box_2d": [ymin, xmin, ymax, xmax],
        "confidence": float(raw_score) if raw_score is not None else 0.0,
    }


def _draw_overlay(display_rgb, boxes: List[Dict[str, Any]], out_path: str) -> str:
    """Render `display_rgb` with each box_2d drawn on it, saved to `out_path`.
    Boxes are in this module's [ymin, xmin, ymax, xmax] pixel-coordinate
    convention. Always writes a file, even with zero boxes (R7: grounding
    must be visibly reachable)."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(display_rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        ymin, xmin, ymax, xmax = box["box_2d"]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0), width=3)
        caption = f"{box['label']} ({box['confidence']:.2f})"
        text_y = max(0, ymin - 12)
        draw.rectangle([xmin, text_y, xmin + 8 * len(caption), text_y + 12], fill=(255, 0, 0))
        draw.text((xmin + 1, text_y), caption, fill=(255, 255, 255))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return os.path.abspath(out_path)


def run_grounding(image_path: str, target_text: str) -> GroundingResult:
    """Localize instances of a text-described target. PLAN.md §4.1."""
    import torch

    raster = load_raster(image_path)
    display_rgb = raster.display_rgb
    height, width = display_rgb.shape[0], display_rgb.shape[1]

    _ensure_fp32_grounding_role()

    from PIL import Image

    pil_image = Image.fromarray(display_rgb).convert("RGB")
    query = _normalize_query(target_text)

    with model_pool.using("grounding") as (model, processor):
        inputs = processor(images=pil_image, text=query, return_tensors="pt")
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

        with torch.no_grad():
            outputs = model(**inputs)

        raw_results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=_DETECTION_THRESHOLD,
            text_threshold=_TEXT_THRESHOLD,
            target_sizes=[(height, width)],
        )[0]

    fallback_label = target_text.strip() or "object"
    scores = raw_results.get("scores", [])
    boxes_raw = raw_results.get("boxes", [])
    labels_raw = raw_results.get("text_labels") or raw_results.get("labels") or []

    bounding_boxes: List[Dict[str, Any]] = []
    for i, box in enumerate(boxes_raw):
        raw_label = labels_raw[i] if i < len(labels_raw) else None
        raw_score = scores[i] if i < len(scores) else 0.0
        bounding_boxes.append(_to_contract_box(box, raw_label, raw_score, fallback_label))

    overlay_name = f"{uuid.uuid4().hex}.png"
    overlay_path = _draw_overlay(
        display_rgb, bounding_boxes, os.path.join(_OVERLAY_DIR, overlay_name)
    )

    if bounding_boxes:
        best = max(bounding_boxes, key=lambda b: b["confidence"])
        text_response = (
            f"Detected {len(bounding_boxes)} instance(s) of '{fallback_label}'. "
            f"Highest-confidence detection scores {best['confidence']:.2f}."
        )
        confidence = float(best["confidence"])
    else:
        text_response = (
            f"No '{fallback_label}' detected above the confidence threshold "
            f"({_DETECTION_THRESHOLD})."
        )
        confidence = 0.0

    result: GroundingResult = {
        "text_response": text_response,
        "bounding_boxes": bounding_boxes,
        "overlay_path": overlay_path,
        "confidence": confidence,
        # "heuristic", not "model_logprob": this is Grounding DINO's own raw
        # per-detection sigmoid score, taken directly with no rescaling. It
        # is model-native (not hand-rolled), but "model_logprob" in this
        # contract specifically denotes a *mean token log-probability* from
        # text generation (what run_vqa/run_caption use) -- a technique this
        # module does not implement, so claiming it here would mislabel the
        # technique (PLAN.md §5.9). "heuristic" is the closest honest fit
        # among the four allowed values; see docs/status/W3.md for the exact
        # computation disclosed.
        "confidence_basis": "heuristic",
    }
    return validate_grounding_result(result)  # type: ignore[return-value]
