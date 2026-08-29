"""
Cross-modal optical-SAR analysis specialist (two MODALITIES, same date/area).

STATUS: STUB (created by W0). **OWNED BY W5** — do not change the signature or
key set without PLAN.md §5.5.

Contract: PLAN.md §4.1 / satquery.contracts.FusionResult

W5 implementation notes (PLAN.md §2.3, §2.4, §4.4):
  - This is the highest-risk rubric row: the hidden eval set is real Cartosat-2S
    (1-band PAN / 4-band MSI) + RISAT (1-2 band) pairs.
  - The fabricated "SAR proxy" in old_files/BT_CM.generate_visual_modalities is
    a cv2.threshold on the OPTICAL image. It is DELETED, not ported. It will
    fail instantly against real radar input.
  - Refuse (via the controller's validation.errors) if both inputs are tagged
    optical. Do not silently proceed — R2 requires two genuinely distinct
    modalities.
  - Real SAR preprocessing: dB/log scaling is MANDATORY and already lives in
    satquery.io.raster. Speckle filtering (Lee / Refined Lee) only if time
    allows — and if you do not implement Lee, do not mention Lee anywhere
    (PLAN.md §2.4, §5.9).
  - Embed optical through benclip's S2 path and SAR through its S1 path; emit a
    per-class agreement/disagreement summary into `evidence`.
  - Keep a SAR-physics fallback that stands alone if benclip underperforms out
    of domain: low backscatter -> smooth water, high backscatter / double-bounce
    -> built-up. Simple, physically defensible, degrades gracefully.
"""

from __future__ import annotations

from satquery.contracts import FusionResult

_STUB_NOTICE = "[stub] Specialist not yet implemented (W0 placeholder; W5 owns this)."


def run_fusion(optical_path: str, sar_path: str, query: str) -> FusionResult:
    """Extract complementary information from a co-registered optical+SAR pair. PLAN.md §4.1."""
    return {
        "text_response": f"{_STUB_NOTICE} query={query!r} optical={optical_path!r} sar={sar_path!r}",
        "agreement_map_path": None,
        "overlay_path": None,
        "evidence": {},
        "confidence": 0.0,
        "confidence_basis": "stub",
    }
