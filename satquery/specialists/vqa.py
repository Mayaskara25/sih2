"""
Single-image VQA and captioning specialists.

STATUS: STUB (created by W0). **OWNED BY W3** — W3 replaces the bodies below with
real Qwen2-VL inference. Do not change the signatures or the returned key sets
without following PLAN.md §5.5; W6's controller and W7's app are already built
against them.

Contract: PLAN.md §4.1 / satquery.contracts.{VQAResult,CaptionResult}

W3 implementation notes (PLAN.md §4.3, §5.9):
  - Acquire the model via satquery.runtime.modelpool. NEVER cache a model in a
    module global here — the card is 3.9 GB and two resident models OOM.
  - Inject benclip.predict_labels() output into the prompt AND into `evidence`;
    that is what makes the answer evidence-grounded rather than a VLM guess.
  - Coerce every number with float() before returning (PLAN.md §4.5) — the
    validators reject numpy scalars because json.dump does too.
  - Set confidence_basis honestly: "model_logprob" if you use mean token
    log-prob, "heuristic" if it is a hand-rolled rule. Never "calibrated"
    unless it was actually calibrated against held-out correctness.
"""

from __future__ import annotations

from typing import Optional

from satquery.contracts import CaptionResult, VQAResult

_STUB_NOTICE = "[stub] Specialist not yet implemented (W0 placeholder; W3 owns this)."


def run_vqa(
    image_path: str,
    query: str,
    *,
    evidence: Optional[dict] = None,
) -> VQAResult:
    """Answer a natural-language question about a single image. PLAN.md §4.1."""
    return {
        "text_response": f"{_STUB_NOTICE} query={query!r} image={image_path!r}",
        "confidence": 0.0,
        "confidence_basis": "stub",
        "evidence": dict(evidence) if evidence else {},
    }


def run_caption(
    image_path: str,
    *,
    evidence: Optional[dict] = None,
) -> CaptionResult:
    """Describe the scene / land cover of a single image. PLAN.md §4.1."""
    return {
        "text_response": f"{_STUB_NOTICE} image={image_path!r}",
        "confidence": 0.0,
        "confidence_basis": "stub",
        "evidence": dict(evidence) if evidence else {},
    }
