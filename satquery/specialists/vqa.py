"""
Single-image VQA and captioning specialists.

STATUS: REAL (W3). Qwen2-VL-2B-Instruct via satquery.runtime.modelpool, 4-bit
NF4 on CUDA (fp32 on CPU) -- exactly the loader W0 wired in modelpool.py's
DEFAULT_REGISTRY["vqa"]; unlike "grounding" this loader needed no correction.

Contract: PLAN.md §4.1 / satquery.contracts.{VQAResult,CaptionResult}

Design notes:
  - The model is acquired via `satquery.runtime.modelpool.model_pool.using
    ("vqa")` and never cached in a module global (PLAN.md §4.3).
  - `benclip.predict_labels()` output (if available) is folded into the
    prompt as grounded evidence AND returned verbatim (label/score list plus
    the §4.5 band_mapping) in the `evidence` dict -- PLAN.md §3.3. The import
    is deferred to call time and wrapped defensively: W2's checkpoint does
    not exist yet at the time this module is written, so `predict_labels`
    raises FileNotFoundError, which is caught, warned once, and treated as
    "no evidence available" rather than a hard failure.
  - If a caller supplies `evidence=` explicitly, that is used as-is (no
    benclip call) -- this preserves the original stub's passthrough contract
    and lets a future caller (or a test) inject evidence directly.
  - `confidence_basis`: "model_logprob", computed as the geometric mean of
    the generated answer's per-token probabilities
    (`exp(mean(transition_log_probs))`) via
    `GenerationMixin.compute_transition_scores(..., normalize_logits=True)`.
    This is a real property of the model's own output distribution, not a
    hand-rolled rule -- see docs/status/W3.md for the exact call. Falls back
    to `"heuristic"` (documented inline) only if score extraction genuinely
    fails (e.g. an empty generation), never silently mislabeled.
  - RSVQA-LR answers are drawn from a tight vocabulary (yes/no, rural/urban,
    bare integers) -- the VQA prompt explicitly asks for a short answer in
    that style; BLEU/accuracy scoring rewards brevity (PLAN.md §5.1 R8 note).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional, Tuple

from satquery.contracts import CaptionResult, VQAResult, validate_caption_result, validate_vqa_result
from satquery.io.raster import RasterInput, load_raster
from satquery.runtime.modelpool import model_pool

_MAX_NEW_TOKENS_VQA = 24
_MAX_NEW_TOKENS_CAPTION = 96

_SYSTEM_PROMPT = (
    "You are SatQuery, a remote-sensing vision assistant. You analyse "
    "single-frame optical, multispectral, and SAR satellite/aerial imagery. "
    "Base your answer only on what is visible in the image (and any land-cover "
    "evidence given below)."
)

_benclip_warned = False


def _warn_benclip_unavailable(reason: str) -> None:
    global _benclip_warned
    if not _benclip_warned:
        warnings.warn(
            f"satquery.specialists.vqa: benclip evidence unavailable ({reason}); "
            "proceeding with an empty evidence dict. This is expected while W2's "
            "checkpoint has not landed (PLAN.md §3.3) and is not a hard failure.",
            RuntimeWarning,
            stacklevel=2,
        )
        _benclip_warned = True


def _benclip_evidence(raster: RasterInput) -> Dict[str, Any]:
    """Best-effort `benclip.predict_labels(raster)` call, per PLAN.md §3.3.

    Imported at call time (not at module import time) so this module never
    breaks if `satquery/adapters/benclip.py` is mid-edit, and defensively
    guarded because W2's checkpoint does not exist yet -- `predict_labels`
    raises FileNotFoundError in that case (see benclip.py::load_benclip).
    Returns {} on any failure, after warning once.
    """
    # Co-residency guard (opt-in via SATQUERY_CO_RESIDENT_MODELS=1). When the
    # pool is holding two heavy models, loading benclip on top of them OOMs the
    # 3.64 GiB card during grounding's forward pass. Skip it -- but SAY SO. An
    # empty evidence dict here is indistinguishable in the trace from "benclip
    # was never used", and the trace is a graded rubric row (R5), so a silent
    # skip would misreport the system's behaviour (PLAN.md §5.9).
    #
    # This lives here, not in modelpool.py: the specialist asks the runtime
    # about residency, never the reverse.
    try:
        from satquery.runtime.modelpool import co_resident_roles, model_pool

        _co = co_resident_roles()
        # resident_roles is a PROPERTY, not a method -- calling it raises
        # TypeError, which the except below would have swallowed, silently
        # disabling this guard. Verified against modelpool.py, not assumed.
        heavies = [r for r in model_pool.resident_roles if r in _co]
        if len(heavies) >= 2:
            return {
                "benclip_labels": [],
                "benclip_skipped": (
                    "benclip evidence was skipped to fit "
                    f"{'+'.join(sorted(heavies))} co-resident on a 3.64 GiB card; "
                    "loading it here OOMs during grounding's forward pass. "
                    "The answer is still produced by the VLM, but carries no "
                    "land-cover labels. Unset SATQUERY_CO_RESIDENT_MODELS to "
                    "restore evidence at the cost of a ~20 s model reload per "
                    "caption/grounding switch."
                ),
            }
    except ImportError:  # pool genuinely unavailable -> fall through
        pass
    # NOTE: deliberately NOT a bare `except Exception`. A broad catch here
    # would hide a bug in this guard (it already hid a TypeError once) and
    # silently reintroduce the OOM it exists to prevent.

    try:
        from satquery.adapters.benclip import predict_labels
    except Exception as exc:  # ImportError, or a mid-edit syntax error, etc.
        _warn_benclip_unavailable(f"import failed: {exc}")
        return {}

    try:
        raw = predict_labels(raster)
    except Exception as exc:  # FileNotFoundError (no checkpoint) is expected
        _warn_benclip_unavailable(f"predict_labels failed: {exc}")
        return {}

    labels = [
        {"label": str(item["label"]), "score": float(item["score"])}
        for item in raw.get("labels", [])
    ]
    band_mapping = raw.get("band_mapping", {})
    return {"benclip_labels": labels, "band_mapping": band_mapping}


def _format_evidence_for_prompt(evidence: Dict[str, Any]) -> str:
    labels = evidence.get("benclip_labels")
    if not labels:
        return ""
    top = sorted(labels, key=lambda item: item.get("score", 0.0), reverse=True)[:5]
    listing = ", ".join(f"{item['label']} ({item['score']:.2f})" for item in top)
    return (
        "\nLand-cover evidence from an RS-adapted encoder (highest similarity "
        f"first, not ground truth): {listing}."
    )


def _build_vqa_prompt(query: str, evidence: Dict[str, Any]) -> str:
    evidence_block = _format_evidence_for_prompt(evidence)
    return (
        f"{_SYSTEM_PROMPT}{evidence_block}\n\n"
        "Answer the following question about this satellite image. Give the "
        "shortest possible correct answer -- a single word, a short phrase, "
        "or a bare number (e.g. 'yes', 'no', 'urban', 'rural', '3'). Do not "
        "explain your reasoning.\n\n"
        f"Question: {query}\nAnswer:"
    )


def _build_caption_prompt(evidence: Dict[str, Any]) -> str:
    evidence_block = _format_evidence_for_prompt(evidence)
    return (
        f"{_SYSTEM_PROMPT}{evidence_block}\n\n"
        "Describe this satellite image in one or two concise sentences: the "
        "dominant land cover / land use and the most salient visible "
        "features. Do not speculate about anything outside the frame."
    )


def _load_image(image_path: str) -> Tuple[RasterInput, Any]:
    from PIL import Image

    raster = load_raster(image_path)
    pil_image = Image.fromarray(raster.display_rgb).convert("RGB")
    return raster, pil_image


def _generate(prompt: str, pil_image: Any, max_new_tokens: int) -> Tuple[str, float, str]:
    """Run one Qwen2-VL generation. Returns (text, confidence, confidence_basis)."""
    import torch

    with model_pool.using("vqa") as (model, processor):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[chat_text], images=[pil_image], return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = gen.sequences[:, input_len:]
        text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        confidence, basis = _confidence_from_scores(model, gen.sequences, gen.scores, input_len)

    return text, confidence, basis


def _confidence_from_scores(model: Any, sequences, scores, input_len: int) -> Tuple[float, str]:
    """Mean per-token probability of the generated answer, as a [0,1] scalar.

    `compute_transition_scores(..., normalize_logits=True)` returns per-step
    log-probabilities of the tokens actually generated. The geometric mean of
    those probabilities (`exp(mean(log_probs))`) is the confidence value;
    labelled "model_logprob" because it is derived directly from the model's
    own token-level probabilities, not a hand-rolled rule (PLAN.md §5.9).
    """
    import torch

    if not scores:
        return 0.0, "heuristic"

    try:
        transition_scores = model.compute_transition_scores(
            sequences, scores, normalize_logits=True
        )
    except Exception:
        return 0.0, "heuristic"

    generated_len = sequences.shape[1] - input_len
    step_scores = transition_scores[0, :generated_len]
    finite = step_scores[torch.isfinite(step_scores)]
    if finite.numel() == 0:
        return 0.0, "heuristic"

    mean_log_prob = float(finite.mean().item())
    confidence = float(min(1.0, max(0.0, math.exp(mean_log_prob))))
    return confidence, "model_logprob"


def run_vqa(
    image_path: str,
    query: str,
    *,
    evidence: Optional[dict] = None,
) -> VQAResult:
    """Answer a natural-language question about a single image. PLAN.md §4.1."""
    raster, pil_image = _load_image(image_path)
    used_evidence = dict(evidence) if evidence else _benclip_evidence(raster)

    prompt = _build_vqa_prompt(query, used_evidence)
    text, confidence, basis = _generate(prompt, pil_image, _MAX_NEW_TOKENS_VQA)

    if not text:
        text = "The model produced no answer for this image."
        confidence, basis = 0.0, "heuristic"

    result: VQAResult = {
        "text_response": text,
        "confidence": float(confidence),
        "confidence_basis": basis,
        "evidence": used_evidence,
    }
    return validate_vqa_result(result)  # type: ignore[return-value]


def run_caption(
    image_path: str,
    *,
    evidence: Optional[dict] = None,
) -> CaptionResult:
    """Describe the scene / land cover of a single image. PLAN.md §4.1."""
    raster, pil_image = _load_image(image_path)
    used_evidence = dict(evidence) if evidence else _benclip_evidence(raster)

    prompt = _build_caption_prompt(used_evidence)
    text, confidence, basis = _generate(prompt, pil_image, _MAX_NEW_TOKENS_CAPTION)

    if not text:
        text = "The model produced no caption for this image."
        confidence, basis = 0.0, "heuristic"

    result: CaptionResult = {
        "text_response": text,
        "confidence": float(confidence),
        "confidence_basis": basis,
        "evidence": used_evidence,
    }
    return validate_caption_result(result)  # type: ignore[return-value]
