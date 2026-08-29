"""
W0 stub conformance: every specialist stub must satisfy its PLAN.md §4.1 contract.

This is the test that makes the stub-first rule (PLAN.md §5.2) safe: W6's
controller and W7's app build against these stubs, so the stubs must be
contract-valid from day one. When W3/W4/W5 replace a stub body, these tests keep
passing unchanged — that is the point.
"""

import pytest

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


def test_run_vqa_stub_is_contract_valid():
    assert validate_vqa_result(run_vqa("img.tif", "what is here?")) is not None


def test_run_vqa_stub_passes_evidence_through():
    result = run_vqa("img.tif", "q", evidence={"labels": ["water"]})
    validate_vqa_result(result)
    assert result["evidence"] == {"labels": ["water"]}


def test_run_caption_stub_is_contract_valid():
    assert validate_caption_result(run_caption("img.tif")) is not None


def test_run_grounding_stub_is_contract_valid():
    assert validate_grounding_result(run_grounding("img.tif", "water body")) is not None


def test_run_change_stub_is_contract_valid():
    assert validate_change_result(run_change("t0.tif", "t1.tif", "what changed?")) is not None


def test_run_fusion_stub_is_contract_valid():
    assert validate_fusion_result(run_fusion("opt.tif", "sar.tif", "find water")) is not None


@pytest.mark.parametrize(
    "fn,args",
    [
        (run_vqa, ("i.tif", "q")),
        (run_caption, ("i.tif",)),
        (run_grounding, ("i.tif", "t")),
        (run_change, ("a.tif", "b.tif", "q")),
        (run_fusion, ("o.tif", "s.tif", "q")),
    ],
)
def test_stubs_declare_themselves_as_stubs(fn, args):
    """confidence_basis must be 'stub' — PLAN.md §5.9 forbids passing a
    placeholder off as a real measurement."""
    assert fn(*args)["confidence_basis"] == "stub"


def test_validators_actually_reject_a_broken_payload():
    """Guards against a validator that accepts anything, which would make every
    other assertion in this file vacuous."""
    broken = dict(run_vqa("i.tif", "q"))
    broken["confidence"] = 1.5
    with pytest.raises(ContractViolation):
        validate_vqa_result(broken)
