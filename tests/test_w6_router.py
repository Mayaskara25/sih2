"""
W6 router tests (PLAN.md §3.4 two-tier routing, §5.4 R4: routing is driven by
quant text, not by how many images were uploaded).

The routing testset in eval/routing_testset.json is the W6 acceptance artefact:
100 hand-labelled queries (all five PS representative queries verbatim) scored
at >= 90% accuracy, with per-task confusion-matrix output for the audit trail.
"""

import json
import os

import pytest

from satquery.controller.router import (
    SINGLE_IMAGE_TASKS,
    RouteResult,
    Tier1Rule,
    extract_grounding_target,
    recommended_input_counts,
    route,
    route_rules,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTSET_PATH = os.path.join(REPO_ROOT, "eval", "routing_testset.json")

PS_QUERIES = [
    ("Describe the land-cover and major objects visible in this image.", "caption"),
    ("Highlight the water body referred to in the query.", "grounding"),
    ("What changed between these two dates, and where did the change occur?", "change"),
    (
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
        "fusion",
    ),
    ("Has the built-up area increased, decreased, or remained unchanged?", "change"),
]

TASKS = ("vqa", "caption", "grounding", "change", "fusion")


# --------------------------------------------------------------------------- #
# Unit behaviour
# --------------------------------------------------------------------------- #


def test_mechanism_is_contract_limited():
    for q in ("what changed?", "describe the scene", "how many boats are here"):
        mech = route(q, n_inputs=2).routing["mechanism"]
        assert mech in {"rule", "exemplar_nn"}


def test_route_returns_route_result():
    r = route("describe the scene", n_inputs=2)
    assert isinstance(r, RouteResult)
    assert r.task in TASKS
    assert isinstance(r.routing["score"], float)
    assert isinstance(r.routing["alternatives_considered"], list)


def test_all_rules_route_to_known_tasks_and_strength_in_range():
    for rule in route_rules():
        assert isinstance(rule, Tier1Rule)
        assert rule.task in TASKS
        assert rule.name
        assert 0.0 < rule.strength <= 1.0


def test_default_fallback_is_vqa():
    assert route("zzz quark plumbus").task == "vqa"


def test_empty_query_falls_back_to_vqa():
    assert route("").task == "vqa"


# --------------------------------------------------------------------------- #
# R4: input count never chooses the task
# --------------------------------------------------------------------------- #


def test_change_query_stays_change_even_with_one_input():
    assert route("what changed?", n_inputs=1).task == "change"


def test_fusion_query_stays_fusion_even_with_two_optical_inputs():
    assert route("use optical and sar images together", n_inputs=2).task == "fusion"


@pytest.mark.parametrize("n", [None, 1, 2])
def test_routing_is_input_count_independent(n):
    for q, gold in ("describe the image", "caption"), ("where is the river", "grounding"):
        assert route(q, n_inputs=n).task == gold


# --------------------------------------------------------------------------- #
# Single-image task image selection (extras the W8 harness grades)
# --------------------------------------------------------------------------- #


def test_two_inputs_single_image_task_defaults_to_image_one():
    r = route("describe the scene you see", n_inputs=2)
    assert r.task == "caption"
    assert r.selected_image_index == 0
    assert r.image_reference == "unspecified"


def test_second_image_request_selects_index_one():
    r = route("tell me about the second image", n_inputs=2)
    assert r.task == "caption"
    assert r.selected_image_index == 1
    assert r.image_reference == "second"


def test_first_image_request_selects_index_zero():
    r = route("describe the first image in this pair", n_inputs=2)
    assert r.selected_image_index == 0


def test_single_image_task_with_two_inputs_is_audited_in_note():
    r = route("highlight the water body", n_inputs=2)
    assert r.task == "grounding"
    assert "single-image task" in r.resolution_note


def test_change_task_needs_two_inputs_but_is_still_chosen():
    r = route("what changed?", n_inputs=1)
    assert r.task == "change"
    assert "validation will refuse" in r.resolution_note


# --------------------------------------------------------------------------- #
# Referring-expression extraction
# --------------------------------------------------------------------------- #


def test_extract_grounding_target_ps_query():
    assert extract_grounding_target("Highlight the water body referred to in the query.") == (
        "water body"
    )


def test_extract_grounding_target_varies():
    assert extract_grounding_target("Where is the river?") == "river"
    assert extract_grounding_target("locate the buildings in the image") == "buildings"
    assert extract_grounding_target("show me where the forest is") == "forest"


def test_grounding_query_routes_to_grounding():
    assert route("Highlight the water body referred to in the query.").task == "grounding"


# --------------------------------------------------------------------------- #
# Acceptance: routing testset >= 90% with confusion matrix
# --------------------------------------------------------------------------- #


def test_testset_exists_and_is_complete():
    assert os.path.exists(TESTSET_PATH), "eval/routing_testset.json missing"
    with open(TESTSET_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    assert len(data) >= 100
    queries = [e["query"] for e in data]
    for q, _ in PS_QUERIES:
        assert q in queries, f"PS representative query must be in the testset: {q!r}"
    for e in data:
        assert e["gold_task"] in TASKS
        assert isinstance(e["input_count"], int)
        # adversarial rows must still have a routing-consistent gold task
        assert e["gold_task"] in TASKS


def test_routing_accuracy_on_testset(capsys):
    with open(TESTSET_PATH, encoding="utf-8") as fh:
        data = json.load(fh)

    conf = {gold: {pred: 0 for pred in TASKS} for gold in TASKS}
    correct = 0
    wrong = []
    for e in data:
        r = route(e["query"], n_inputs=e["input_count"])
        conf[e["gold_task"]][r.task] += 1
        if r.task == e["gold_task"]:
            correct += 1
        else:
            wrong.append(e["id"])

    acc = correct / len(data)

    header = "            " + "".join(f"{t:>9}" for t in TASKS)
    rows = [header]
    for gold in TASKS:
        row = f"{gold:>9}"
        for pred in TASKS:
            row += f"{conf[gold][pred]:>9}"
        rows.append(row)
    with capsys.disabled():
        print("\n".join(rows) + f"\naccuracy={acc:.3f} ({correct}/{len(data)}); wrong={wrong}")

    assert acc >= 0.90, f"routing accuracy {acc:.3f} below 90% acceptance floor"


# --------------------------------------------------------------------------- #
# Helper sanity: recommended_input_counts feeds validate.py
# --------------------------------------------------------------------------- #


def test_recommended_input_counts():
    assert recommended_input_counts("change") == (2, 2)
    assert recommended_input_counts("fusion") == (2, 2)
    for task in SINGLE_IMAGE_TASKS:
        assert recommended_input_counts(task) == (1, 2)