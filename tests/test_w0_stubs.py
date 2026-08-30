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


def test_run_fusion_stub_is_contract_valid():
    assert validate_fusion_result(run_fusion("opt.tif", "sar.tif", "find water")) is not None


@pytest.mark.parametrize(
    "fn,args",
    [(run_change, ("a.tif", "b.tif", "q")), (run_fusion, ("o.tif", "s.tif", "q"))],
)
def test_unimplemented_specialists_declare_themselves_as_stubs(fn, args):
    """PLAN.md §5.9: a placeholder must announce itself rather than pass as a
    real measurement. Delete a case here only when that specialist becomes real."""
    assert fn(*args)["confidence_basis"] == "stub"


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
