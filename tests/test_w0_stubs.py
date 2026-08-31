"""
Specialist contract conformance (PLAN.md §4.1).

Originally this file asserted that every specialist was a *stub*. That was a
design error on W0's part: PLAN.md §5.2 promises "when a real implementation
lands, it replaces the stub body and nothing downstream changes" — but a test
asserting `confidence_basis == "stub"` breaks by construction the moment the
promise is kept. W3 landing real `run_vqa`/`run_caption`/`run_grounding` broke
8 tests here for exactly that reason.

The durable property is **contract conformance**, which holds for a stub and a
real implementation alike. That is what this file tests now.

Two consequences worth keeping in mind:
  - Still-stubbed specialists (`run_change`, `run_fusion`) are tested with fake
    paths, because a stub never opens the file. Real specialists must be given
    a REAL image, and are gated behind CUDA + data being present so a clone
    without a GPU (or without the ~5 GB of weights) still gets a green suite.
  - When W4/W5 land, move `run_change`/`run_fusion` from the stub section to
    the real section rather than deleting their assertions.
"""

import glob

import pytest
import torch

from satquery.contracts import (
    ContractViolation,
    validate_caption_result,
    validate_change_result,
    validate_fusion_result,
    validate_grounding_result,
    validate_vqa_result,
)
from satquery.specialists.change import run_change
from satquery.specialists.fusion import run_fusion
from satquery.specialists.grounding import run_grounding
from satquery.specialists.vqa import run_caption, run_vqa

CUDA_AVAILABLE = torch.cuda.is_available()
_RSVQA = sorted(glob.glob("data/rsvqa_lr/Images_LR/*.tif"))
_real_image = _RSVQA[0] if _RSVQA else None

requires_model = pytest.mark.skipif(
    not CUDA_AVAILABLE or _real_image is None,
    reason="requires a CUDA device and RSVQA-LR data on disk",
)


# ---------------------------------------------------------------------------
# Still stubbed (W4/W5 not landed): fake paths are fine, a stub never opens one.
# ---------------------------------------------------------------------------


def test_run_change_stub_is_contract_valid():
    assert validate_change_result(run_change("t0.tif", "t1.tif", "what changed?")) is not None



def test_run_change_stub_declares_itself_as_stub():
    """PLAN.md §5.9: a placeholder must announce itself rather than pass as a
    real measurement. Delete this when W4 lands run_change."""
    assert run_change("a.tif", "b.tif", "q")["confidence_basis"] == "stub"


# ---------------------------------------------------------------------------
# Real implementations (W3): need a real image, gated so a GPU-less clone passes.
# ---------------------------------------------------------------------------


@requires_model
def test_run_vqa_is_contract_valid():
    assert validate_vqa_result(run_vqa(_real_image, "what land cover is shown?")) is not None


@requires_model
def test_run_caption_is_contract_valid():
    assert validate_caption_result(run_caption(_real_image)) is not None


@requires_model
def test_run_grounding_is_contract_valid():
    assert validate_grounding_result(run_grounding(_real_image, "water body")) is not None


@requires_model
def test_real_specialists_do_not_claim_to_be_stubs():
    """The inverse of the stub check: once implemented, a specialist must stop
    reporting "stub" — otherwise W7's app would keep rendering a real answer as
    "Placeholder — specialist not yet implemented"."""
    assert run_vqa(_real_image, "what is here?")["confidence_basis"] != "stub"


@requires_model
def test_run_vqa_passes_evidence_through():
    result = run_vqa(_real_image, "q", evidence={"labels": ["water"]})
    validate_vqa_result(result)
    assert result["evidence"].get("labels") == ["water"]


# Real implementation (W5): run_fusion is now live, not a stub.
# Needs BEN S1+S2 data (gated behind _real_image presence).


def test_run_fusion_is_contract_valid():
    """W5: run_fusion now returns a real result. Moved from the stub section
    per W5 brief line 81 — the stub assertion no longer applies."""
    if _real_image is None:
        pytest.skip("no BEN data on disk")
    # Use any available S1+S2 pair — the contract validator does not care
    # about the specific content, only the shape.
    import glob as _glob
    s1_files = _glob.glob("data/bigearthnet/images/BigEarthNet-S1/**/*_VV.tif",
                          recursive=True)
    if not s1_files:
        pytest.skip("no S1 data on disk")
    result = run_fusion(_real_image, s1_files[0], "what land cover?")
    assert validate_fusion_result(result) is not None
    assert result["confidence_basis"] != "stub"


# ---------------------------------------------------------------------------
# Guard against a validator that rubber-stamps anything, which would make every
# assertion above vacuous. Uses a stub so it needs no GPU and no weights.
# ---------------------------------------------------------------------------


def test_validators_actually_reject_a_broken_payload():
    broken = dict(run_change("a.tif", "b.tif", "q"))
    broken["confidence"] = 1.5
    with pytest.raises(ContractViolation):
        validate_change_result(broken)


def test_validators_reject_an_unknown_confidence_basis():
    broken = dict(run_fusion("o.tif", "s.tif", "q"))
    broken["confidence_basis"] = "vibes"
    with pytest.raises(ContractViolation):
        validate_fusion_result(broken)
