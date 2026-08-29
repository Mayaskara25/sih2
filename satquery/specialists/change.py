"""
Bi-temporal change analysis specialist (two dates, SAME modality).

STATUS: STUB (created by W0). **OWNED BY W4** — do not change the signature or
key set without PLAN.md §5.5.

Contract: PLAN.md §4.1 / satquery.contracts.ChangeResult

W4 implementation notes (PLAN.md §2.5, §4.1):
  - This path must stay DISTINCT from fusion.py (R3). Two dates of one modality
    here; two modalities here-and-now in fusion.py.
  - If you adapt old_files/BT_CM.py, EVERY §2.5 bug must be fixed first:
      * fit the affine on the FILTERED match list, not the unfiltered one
      * null-check cv2.estimateAffinePartial2D — it returns None on low-feature
        scenes (water, farmland) and the next line crashes
      * guard orb.detectAndCompute returning empty descriptors
    Prefer skipping registration entirely when both inputs are georeferenced and
    already co-registered — check crs/transform from satquery.io.raster first.
  - What makes this REAL and not prompt-stuffing: run benclip.predict_labels()
    on BOTH dates and emit a structured per-class delta into `evidence`. The
    answer is then grounded in an RS-adapted model's read of both scenes.
  - metrics should carry at least registration_confidence and
    changed_area_fraction. Coerce all numbers with float() (PLAN.md §4.5).
"""

from __future__ import annotations

from satquery.contracts import ChangeResult

_STUB_NOTICE = "[stub] Specialist not yet implemented (W0 placeholder; W4 owns this)."


def run_change(image_path_t0: str, image_path_t1: str, query: str) -> ChangeResult:
    """Analyse change between two dates of the same modality. PLAN.md §4.1."""
    return {
        "text_response": f"{_STUB_NOTICE} query={query!r} t0={image_path_t0!r} t1={image_path_t1!r}",
        "change_mask_path": None,
        "overlay_path": None,
        "metrics": {},
        "confidence": 0.0,
        "confidence_basis": "stub",
        "evidence": {},
    }
