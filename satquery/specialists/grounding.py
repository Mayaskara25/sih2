"""
Text-guided visual grounding specialist.

STATUS: STUB (created by W0). **OWNED BY W3** — W3 replaces the body with real
Grounding DINO inference. Do not change the signature or key set without
PLAN.md §5.5.

Contract: PLAN.md §4.1 / satquery.contracts.{GroundingResult,BoundingBox}

W3 implementation notes:
  - box_2d is [ymin, xmin, ymax, xmax] in PIXEL coordinates. This order is NOT
    what Grounding DINO returns natively (it gives [xmin, ymin, xmax, ymax]) —
    the old draft got this right, keep it right. The validator enforces
    ymin<=ymax and xmin<=xmax.
  - transformers resolved to 5.16.x here, so post_process_grounded_object_detection
    takes `threshold=`, not the 4.x `box_threshold=`. Do not copy the try/except
    version-straddling hack from old_files/models_registry.py (PLAN.md §2.2a).
  - overlay_path is REQUIRED to be a real rendered image for R7: grounding must
    be visibly reachable from the app, not just return numbers.
  - Acquire the model via satquery.runtime.modelpool; no module-global cache.
"""

from __future__ import annotations

from satquery.contracts import GroundingResult

_STUB_NOTICE = "[stub] Specialist not yet implemented (W0 placeholder; W3 owns this)."


def run_grounding(image_path: str, target_text: str) -> GroundingResult:
    """Localize instances of a text-described target. PLAN.md §4.1."""
    return {
        "text_response": f"{_STUB_NOTICE} target={target_text!r} image={image_path!r}",
        "bounding_boxes": [],
        "overlay_path": None,
        "confidence": 0.0,
        "confidence_basis": "stub",
    }
