"""
W9 router fidelity fix: rural_urban queries must route to vqa.

Acceptance:
  - "Is it a rural or an urban area?" -> vqa (the 100/100 category)
  - The three control queries still route to grounding / fusion / vqa
  - Both tokens required: single-token urban queries must NOT be diverted
  - eval/routing_testset.json unmodified (checked via git status in CI, not here)
  - Real-query fidelity over 10,004 RSVQA questions improves from 222 to 122
"""

import json
import os

import pytest

from satquery.controller.router import route

# ---------------------------------------------------------------------------
# Task 1 core assertions
# ---------------------------------------------------------------------------


def test_rural_urban_query_routes_to_vqa():
    assert route("Is it a rural or an urban area?").task == "vqa"
    # mechanism should be the new Tier-1 rule, not exemplar fallback
    r = route("Is it a rural or an urban area?")
    assert r.routing["mechanism"] == "rule"
    assert r.routing["matched"] == "vqa_rural_urban"


def test_rural_urban_variants_route_to_vqa():
    # order swapped, case insensitive, extra words
    for q in [
        "Is it urban or rural?",
        "RURAL and URBAN classification",
        "is this rural or urban?",
        "Is it a rural or an urban area",
        "rural vs urban area?",
    ]:
        assert route(q).task == "vqa", f"expected vqa for {q!r} got {route(q).task}"


def test_three_control_queries_unchanged():
    # These three are the brief's safety check – widening the rule must not
    # capture legitimate non-VQA queries that already route correctly.
    assert route("find the urban area").task == "grounding"
    assert route("find the urban area").routing["matched"] == "grounding_find_object"
    assert route("use optical together with sar to map the urban extent").task == "fusion"
    assert route("what fraction of the image is urban area").task == "vqa"
    assert route("what fraction of the image is urban area").routing["matched"] == "vqa_what_fraction"


def test_single_token_urban_does_not_match_rural_urban_rule():
    # Neither of the first two control queries contains 'rural', so the paired
    # form cannot touch them; also check that lone urban stays grounding.
    assert route("find the urban area").task == "grounding"
    # single urban with no rural, even with vqa-like wording, should NOT be forced to vqa via rural_urban rule
    # "urban area" alone without rural should route via exemplar or other rule, not rural_urban
    r = route("urban area")
    # It may route to grounding via exemplar, but must NOT be via vqa_rural_urban
    assert r.routing["matched"] != "vqa_rural_urban"


def test_rural_urban_rule_requires_both_tokens():
    # Queries with only one of the two tokens must not hit the new rule
    for q in ["rural area", "urban area", "find the rural area", "is it rural?"]:
        r = route(q)
        assert r.routing["matched"] != "vqa_rural_urban", f"{q!r} unexpectedly matched rural_urban"


def test_grounding_tier1_does_not_fire_on_rural_urban():
    # The question contains none of highlight/locate/localise/where is/find the
    # so no grounding Tier-1 rule should fire; the vqa_rural_urban rule wins.
    r = route("Is it a rural or an urban area?")
    assert r.task == "vqa"
    assert not r.routing["matched"].startswith("grounding")


# ---------------------------------------------------------------------------
# Real-query fidelity (CPU, no GPU)
# ---------------------------------------------------------------------------


def test_real_query_fidelity_improves():
    # Re-measure over the full RSVQA test split (10,004 active questions).
    # This is the W8 method: pre-route every active question and count those
    # not landing on vqa. After the fix, rural_urban 0/100 and total 122/10004.
    qpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rsvqa_lr", "LR_split_test_questions.json")
    if not os.path.exists(qpath):
        pytest.skip("RSVQA data not on disk")
    with open(qpath) as fh:
        data = json.load(fh)
    active = [q for q in data["questions"] if q.get("active")]
    assert len(active) == 10004
    by_type = {}
    for q in active:
        by_type.setdefault(q["type"], []).append(q)
    # rural_urban must be 0 misrouted
    rural = by_type.get("rural_urban", [])
    mis_rural = sum(1 for q in rural if route(q["question"]).task != "vqa")
    assert mis_rural == 0, f"rural_urban misrouted {mis_rural}/100 expected 0"
    total_mis = sum(1 for q in active if route(q["question"]).task != "vqa")
    assert total_mis == 122, f"total misrouted {total_mis}/10004 expected 122 (was 222 before)"
    # fidelity 98.78%
    fidelity = 1 - total_mis / len(active)
    assert fidelity > 0.987
