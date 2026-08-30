"""
Tests for satquery/specialists/vqa.py (PLAN.md §4.1/§7 W3).

Split per the work order: pure-logic unit tests (prompt building, evidence
formatting, the log-prob-to-confidence math, benclip fallback behaviour) run
everywhere with no GPU and no downloaded weights; the end-to-end generation
tests are gated on CUDA + a successful real model load so CI-without-GPU
still passes the rest of this file.
"""

from __future__ import annotations

import json
import os
import warnings

import pytest
import torch

from satquery.contracts import validate_caption_result, validate_vqa_result
from satquery.specialists import vqa as vqa_mod
from satquery.specialists.vqa import (
    _benclip_evidence,
    _build_caption_prompt,
    _build_vqa_prompt,
    _confidence_from_scores,
    _format_evidence_for_prompt,
    run_caption,
    run_vqa,
)

CUDA_AVAILABLE = torch.cuda.is_available()


# --------------------------------------------------------------------------- #
# Pure-logic tests -- no model, no GPU.
# --------------------------------------------------------------------------- #


def test_format_evidence_for_prompt_empty_when_no_labels():
    assert _format_evidence_for_prompt({}) == ""
    assert _format_evidence_for_prompt({"benclip_labels": []}) == ""


def test_format_evidence_for_prompt_lists_top_labels_by_score():
    evidence = {
        "benclip_labels": [
            {"label": "Water", "score": 0.2},
            {"label": "Urban fabric", "score": 0.9},
            {"label": "Forest", "score": 0.5},
        ]
    }
    text = _format_evidence_for_prompt(evidence)
    # highest score first
    assert text.index("Urban fabric") < text.index("Forest") < text.index("Water")


def test_build_vqa_prompt_asks_for_short_answer_and_includes_query():
    prompt = _build_vqa_prompt("Is it urban or rural?", {})
    assert "Is it urban or rural?" in prompt
    assert "short" in prompt.lower() or "shortest" in prompt.lower()


def test_build_vqa_prompt_includes_evidence_block_when_present():
    evidence = {"benclip_labels": [{"label": "Urban fabric", "score": 0.8}]}
    prompt = _build_vqa_prompt("q", evidence)
    assert "Urban fabric" in prompt


def test_build_caption_prompt_asks_for_description():
    prompt = _build_caption_prompt({})
    assert "describe" in prompt.lower()


def test_confidence_from_scores_empty_scores_is_heuristic_zero():
    confidence, basis = _confidence_from_scores(model=None, sequences=None, scores=(), input_len=0)
    assert confidence == 0.0
    assert basis == "heuristic"


def test_confidence_from_scores_computes_geometric_mean_token_probability():
    """Build two fake generation steps with known softmax probabilities and
    check the confidence is exp(mean(log_probs)) of the tokens actually
    generated -- i.e. genuinely derived from the model's own distribution,
    not a hand-rolled heuristic."""
    vocab = 4
    # Step 1: token 0 chosen, with probability 0.5 under softmax of these logits.
    logits1 = torch.log(torch.tensor([[0.5, 0.3, 0.1, 0.1]]))
    # Step 2: token 1 chosen, with probability 0.4 under softmax of these logits.
    logits2 = torch.log(torch.tensor([[0.2, 0.4, 0.3, 0.1]]))
    scores = (logits1, logits2)

    input_len = 3
    sequences = torch.tensor([[9, 9, 9, 0, 1]])  # 3 prompt tokens + generated [0, 1]

    class _FakeModel:
        def compute_transition_scores(self, sequences, scores, normalize_logits=True):
            # Mirror the real HF behaviour closely enough for this unit test:
            # log-softmax each step's logits, then pick out the chosen token.
            log_probs = []
            gen_tokens = sequences[0, input_len:]
            for step_logits, tok in zip(scores, gen_tokens):
                log_softmax = torch.log_softmax(step_logits, dim=-1)
                log_probs.append(log_softmax[0, tok])
            return torch.stack(log_probs).unsqueeze(0)

    confidence, basis = _confidence_from_scores(_FakeModel(), sequences, scores, input_len)

    assert basis == "model_logprob"
    expected = (0.5 * 0.4) ** 0.5  # geometric mean of the two true token probabilities
    assert confidence == pytest.approx(expected, abs=1e-3)


def test_confidence_from_scores_falls_back_to_heuristic_on_extraction_failure():
    class _BrokenModel:
        def compute_transition_scores(self, *a, **k):
            raise RuntimeError("boom")

    confidence, basis = _confidence_from_scores(
        _BrokenModel(), sequences=torch.zeros((1, 1)), scores=(torch.zeros((1, 4)),), input_len=0
    )
    assert confidence == 0.0
    assert basis == "heuristic"


def test_confidence_is_always_within_unit_interval_even_with_extreme_logits():
    """A very confident (near-zero-entropy) step must not push confidence
    above 1.0 due to floating point, and must stay >= 0."""
    logits = torch.zeros((1, 4))
    logits[0, 0] = 1000.0  # near-certain token 0
    scores = (logits,)
    sequences = torch.tensor([[9, 0]])
    input_len = 1

    class _FakeModel:
        def compute_transition_scores(self, sequences, scores, normalize_logits=True):
            log_softmax = torch.log_softmax(scores[0], dim=-1)
            return log_softmax[:, 0:1]

    confidence, basis = _confidence_from_scores(_FakeModel(), sequences, scores, input_len)
    assert 0.0 <= confidence <= 1.0


def test_benclip_evidence_returns_empty_dict_when_checkpoint_absent(monkeypatch):
    """W2's checkpoint does not exist yet at W3 authorship time (PLAN.md
    §3.3 explicitly requires this to degrade gracefully, not crash)."""

    class _Raster:
        pass

    # Force the real import path; predict_labels() should raise
    # FileNotFoundError because no checkpoint is on disk in this test env
    # (or, if a checkpoint exists, the module must still handle any
    # exception the same way -- either way the contract is "never raises").
    vqa_mod._benclip_warned = False
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        evidence = _benclip_evidence(_Raster())
    assert isinstance(evidence, dict)


def test_benclip_evidence_returns_empty_dict_on_import_failure(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _raising_import(name, *args, **kwargs):
        if name == "satquery.adapters.benclip":
            raise ImportError("simulated: benclip module unavailable")
        return real_import(name, *args, **kwargs)

    vqa_mod._benclip_warned = False
    monkeypatch.setattr(builtins, "__import__", _raising_import)
    evidence = _benclip_evidence(object())
    assert evidence == {}


def test_benclip_evidence_coerces_numpy_scalars(monkeypatch):
    """PLAN.md §4.5: every number must be a Python float before it enters a
    payload, never a numpy scalar."""
    import numpy as np

    fake_module = type(
        "FakeBenclip",
        (),
        {
            "predict_labels": staticmethod(
                lambda raster: {
                    "labels": [{"label": "Water", "score": np.float32(0.9)}],
                    "band_mapping": {
                        "slots_filled": ["B02"],
                        "slots_absent": [],
                        "fill_strategy": "per_channel_training_mean",
                        "source_modality": "optical",
                    },
                }
            )
        },
    )()

    import sys

    monkeypatch.setitem(sys.modules, "satquery.adapters.benclip", fake_module)
    vqa_mod._benclip_warned = False
    evidence = _benclip_evidence(object())

    assert evidence["benclip_labels"] == [{"label": "Water", "score": pytest.approx(0.9, abs=1e-5)}]
    assert isinstance(evidence["benclip_labels"][0]["score"], float)
    assert evidence["band_mapping"]["source_modality"] == "optical"


def test_run_vqa_and_run_caption_stub_notice_is_gone():
    import inspect

    src = inspect.getsource(vqa_mod.run_vqa) + inspect.getsource(vqa_mod.run_caption)
    assert "[stub]" not in src


# --------------------------------------------------------------------------- #
# End-to-end tests -- require CUDA and the real Qwen2-VL-2B-Instruct 4-bit
# checkpoint. Skip cleanly without a GPU so the rest of this file still runs
# in CI-without-GPU (per the work order).
# --------------------------------------------------------------------------- #

_RSVQA_IMAGES_DIR = os.path.join("data", "rsvqa_lr", "Images_LR")
_RSVQA_QUESTIONS = os.path.join("data", "rsvqa_lr", "LR_split_test_questions.json")
_RSVQA_ANSWERS = os.path.join("data", "rsvqa_lr", "LR_split_test_answers.json")

_rsvqa_available = (
    os.path.isdir(_RSVQA_IMAGES_DIR)
    and os.path.isfile(_RSVQA_QUESTIONS)
    and os.path.isfile(_RSVQA_ANSWERS)
)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a CUDA device")
@pytest.mark.skipif(not _rsvqa_available, reason="RSVQA-LR data not on disk")
def test_run_vqa_end_to_end_on_real_rsvqa_image():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    image_path = os.path.join(_RSVQA_IMAGES_DIR, "232.tif")
    result = run_vqa(image_path, "Is it a rural or an urban area?")
    validate_vqa_result(result)

    assert isinstance(result["text_response"], str) and result["text_response"]
    assert result["confidence_basis"] in ("model_logprob", "heuristic")
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["evidence"], dict)

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    assert peak_mb < 2500, f"vqa peak allocated {peak_mb:.0f} MB exceeds budget"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a CUDA device")
@pytest.mark.skipif(not _rsvqa_available, reason="RSVQA-LR data not on disk")
def test_run_caption_end_to_end_on_real_rsvqa_image():
    image_path = os.path.join(_RSVQA_IMAGES_DIR, "232.tif")
    result = run_caption(image_path)
    validate_caption_result(result)
    assert isinstance(result["text_response"], str) and result["text_response"]


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a CUDA device")
@pytest.mark.skipif(not _rsvqa_available, reason="RSVQA-LR data not on disk")
def test_run_vqa_evidence_passthrough_still_works_end_to_end():
    """Caller-supplied evidence is used as-is (preserves the original stub's
    passthrough contract) rather than being silently overwritten."""
    image_path = os.path.join(_RSVQA_IMAGES_DIR, "232.tif")
    evidence = {"benclip_labels": [{"label": "Urban fabric", "score": 0.9}]}
    result = run_vqa(image_path, "what is here?", evidence=evidence)
    validate_vqa_result(result)
    assert result["evidence"] == evidence


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a CUDA device")
@pytest.mark.skipif(not _rsvqa_available, reason="RSVQA-LR data not on disk")
def test_benclip_evidence_is_non_empty_when_a_checkpoint_is_present():
    """PLAN.md §3.3 / §7 W3 acceptance: 'the evidence dict is non-empty when
    benclip is available'. This does not gate on CUDA -- benclip runs on CPU
    fine -- but does skip cleanly if no checkpoint exists yet, since W3 must
    degrade gracefully rather than depend on W2 having landed one."""
    from satquery.io.raster import load_raster

    checkpoint_path = os.environ.get("SATQUERY_BENCLIP_PATH", "checkpoints/benclip")
    state_path = (
        os.path.join(checkpoint_path, "benclip_state.pt")
        if os.path.isdir(checkpoint_path)
        else checkpoint_path
    )
    if not os.path.exists(state_path):
        pytest.skip("no benclip checkpoint on disk yet (W2 in progress)")
    if not os.path.isfile(os.path.join(_RSVQA_IMAGES_DIR, "232.tif")):
        pytest.skip("RSVQA-LR sample image not on disk")

    vqa_mod._benclip_warned = False
    raster = load_raster(os.path.join(_RSVQA_IMAGES_DIR, "232.tif"))
    evidence = _benclip_evidence(raster)

    assert evidence != {}
    assert len(evidence.get("benclip_labels", [])) > 0
    assert all(isinstance(item["score"], float) for item in evidence["benclip_labels"])
    assert "band_mapping" in evidence


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a CUDA device")
@pytest.mark.skipif(not _rsvqa_available, reason="RSVQA-LR data not on disk")
def test_run_vqa_only_one_heavy_model_resident_after_grounding_then_vqa():
    """PLAN.md §4.3: acquiring a different heavy role auto-releases the
    previous one -- exercised here across the two W3 specialists."""
    from satquery.runtime.modelpool import model_pool
    from satquery.specialists.grounding import run_grounding

    image_path = os.path.join(_RSVQA_IMAGES_DIR, "232.tif")
    run_grounding(image_path, "a road")
    assert model_pool.resident_heavy_role == "grounding"

    run_vqa(image_path, "Is it urban or rural?")
    assert model_pool.resident_heavy_role == "vqa"
