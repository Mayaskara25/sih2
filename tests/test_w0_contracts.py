"""Tests for satquery/contracts.py — the §4 machine-readable contract copy.

For every validate_* function: one well-formed payload that must pass, and
one payload per failure mode (missing key, extra key, wrong type,
out-of-range confidence, bad confidence_basis, malformed box_2d) that must
raise ContractViolation. Failure-mode tests that don't apply to a given
validator's shape (e.g. box_2d for a text-only result, confidence for
band_mapping/execution_trace) are replaced with the nearest analogous
enum-domain check named in PLAN.md's REJECT list (task_selected,
source_modality, routing.mechanism, inputs[].modality).
"""

import copy

import pytest

from satquery.contracts import (
    BandMapping,
    BoundingBox,
    CaptionResult,
    ChangeResult,
    ContractViolation,
    ExecutionTrace,
    FusionResult,
    GroundingResult,
    LabelPrediction,
    VQAResult,
    validate_band_mapping,
    validate_caption_result,
    validate_change_result,
    validate_execution_trace,
    validate_fusion_result,
    validate_grounding_result,
    validate_vqa_result,
)


# ---------------------------------------------------------------------------
# Well-formed fixtures
# ---------------------------------------------------------------------------


def _valid_vqa_result():
    return {
        "text_response": "There are two buildings visible.",
        "confidence": 0.5,
        "confidence_basis": "stub",
        "evidence": {},
    }


def _valid_caption_result():
    return {
        "text_response": "A satellite image showing farmland.",
        "confidence": 0.42,
        "confidence_basis": "heuristic",
        "evidence": {"labels": ["farmland"]},
    }


def _valid_bounding_box():
    return {"label": "building", "box_2d": [10, 20, 30, 40], "confidence": 0.9}


def _valid_grounding_result():
    return {
        "text_response": "Found 1 building.",
        "bounding_boxes": [_valid_bounding_box()],
        "overlay_path": "runs/abc/overlay.png",
        "confidence": 0.9,
        "confidence_basis": "model_logprob",
    }


def _valid_change_result():
    return {
        "text_response": "Built-up area increased.",
        "change_mask_path": "runs/abc/mask.png",
        "overlay_path": "runs/abc/overlay.png",
        "metrics": {"changed_area_fraction": 0.12},
        "confidence": 0.7,
        "confidence_basis": "calibrated",
        "evidence": {"t0_labels": [], "t1_labels": []},
    }


def _valid_fusion_result():
    return {
        "text_response": "Optical and SAR agree on water extent.",
        "agreement_map_path": "runs/abc/agreement.png",
        "overlay_path": None,
        "evidence": {"optical_labels": [], "sar_labels": []},
        "confidence": 0.65,
        "confidence_basis": "heuristic",
    }


def _valid_execution_trace():
    return {
        "run_id": "run-0001",
        "timestamp": "2026-08-29T12:00:00Z",
        "query": "What changed between these two images?",
        "task_selected": "change",
        "routing": {
            "mechanism": "rule",
            "matched": "change_keyword",
            "score": 1.0,
            "alternatives_considered": ["fusion"],
        },
        "inputs": [
            {
                "path": "data/a.tif",
                "modality": "optical",
                "bands": 3,
                "shape": [512, 512],
                "format": "GeoTIFF",
                "crs": "EPSG:4326",
                "checks_passed": True,
            }
        ],
        "validation": {"passed": True, "warnings": [], "errors": []},
        "models_used": [
            {
                "role": "change_backbone",
                "name": "benclip",
                "revision": "v0",
                "precision": "fp16",
                "adapter": None,
                "device": "cuda",
            }
        ],
        "parameters": {},
        "result": {},
        "artifacts": {"mask": None, "overlay": None, "report": None},
        "timings_ms": {"routing": 1, "validation": 2, "inference": 100, "total": 103},
    }


def _valid_band_mapping():
    return {
        "slots_filled": ["B04", "B03", "B02"],
        "slots_absent": ["B01"],
        "fill_strategy": "per_channel_training_mean",
        "source_modality": "optical",
    }


# ---------------------------------------------------------------------------
# validate_vqa_result
# ---------------------------------------------------------------------------


def test_vqa_result_valid_passes():
    payload = _valid_vqa_result()
    assert validate_vqa_result(payload) == payload


def test_vqa_result_missing_key():
    payload = _valid_vqa_result()
    del payload["evidence"]
    with pytest.raises(ContractViolation):
        validate_vqa_result(payload)


def test_vqa_result_extra_key():
    payload = _valid_vqa_result()
    payload["unexpected"] = "nope"
    with pytest.raises(ContractViolation):
        validate_vqa_result(payload)


def test_vqa_result_wrong_type():
    payload = _valid_vqa_result()
    payload["confidence"] = "high"
    with pytest.raises(ContractViolation):
        validate_vqa_result(payload)


def test_vqa_result_confidence_out_of_range():
    payload = _valid_vqa_result()
    payload["confidence"] = 1.5
    with pytest.raises(ContractViolation):
        validate_vqa_result(payload)


def test_vqa_result_bad_confidence_basis():
    payload = _valid_vqa_result()
    payload["confidence_basis"] = "vibes"
    with pytest.raises(ContractViolation):
        validate_vqa_result(payload)


# ---------------------------------------------------------------------------
# validate_caption_result (identical shape to vqa per §4.1)
# ---------------------------------------------------------------------------


def test_caption_result_valid_passes():
    payload = _valid_caption_result()
    assert validate_caption_result(payload) == payload


def test_caption_result_missing_key():
    payload = _valid_caption_result()
    del payload["text_response"]
    with pytest.raises(ContractViolation):
        validate_caption_result(payload)


def test_caption_result_extra_key():
    payload = _valid_caption_result()
    payload["extra_field"] = 1
    with pytest.raises(ContractViolation):
        validate_caption_result(payload)


def test_caption_result_wrong_type():
    payload = _valid_caption_result()
    payload["evidence"] = "not a dict"
    with pytest.raises(ContractViolation):
        validate_caption_result(payload)


def test_caption_result_confidence_out_of_range():
    payload = _valid_caption_result()
    payload["confidence"] = -0.01
    with pytest.raises(ContractViolation):
        validate_caption_result(payload)


def test_caption_result_bad_confidence_basis():
    payload = _valid_caption_result()
    payload["confidence_basis"] = "definitely_calibrated_trust_me"
    with pytest.raises(ContractViolation):
        validate_caption_result(payload)


# ---------------------------------------------------------------------------
# validate_grounding_result
# ---------------------------------------------------------------------------


def test_grounding_result_valid_passes():
    payload = _valid_grounding_result()
    assert validate_grounding_result(payload) == payload


def test_grounding_result_missing_key():
    payload = _valid_grounding_result()
    del payload["overlay_path"]
    with pytest.raises(ContractViolation):
        validate_grounding_result(payload)


def test_grounding_result_extra_key():
    payload = _valid_grounding_result()
    payload["debug"] = True
    with pytest.raises(ContractViolation):
        validate_grounding_result(payload)


def test_grounding_result_wrong_type():
    payload = _valid_grounding_result()
    payload["bounding_boxes"] = "not-a-list"
    with pytest.raises(ContractViolation):
        validate_grounding_result(payload)


def test_grounding_result_confidence_out_of_range():
    payload = _valid_grounding_result()
    payload["confidence"] = 2.0
    with pytest.raises(ContractViolation):
        validate_grounding_result(payload)


def test_grounding_result_bad_confidence_basis():
    payload = _valid_grounding_result()
    payload["confidence_basis"] = "guessed"
    with pytest.raises(ContractViolation):
        validate_grounding_result(payload)


def test_grounding_result_malformed_box_2d_wrong_length():
    payload = _valid_grounding_result()
    payload["bounding_boxes"] = [
        {"label": "building", "box_2d": [10, 20, 30], "confidence": 0.9}
    ]
    with pytest.raises(ContractViolation):
        validate_grounding_result(payload)


def test_grounding_result_malformed_box_2d_ymin_gt_ymax():
    payload = _valid_grounding_result()
    payload["bounding_boxes"] = [
        {"label": "building", "box_2d": [30, 20, 10, 40], "confidence": 0.9}
    ]
    with pytest.raises(ContractViolation):
        validate_grounding_result(payload)


def test_grounding_result_malformed_box_2d_xmin_gt_xmax():
    payload = _valid_grounding_result()
    payload["bounding_boxes"] = [
        {"label": "building", "box_2d": [10, 40, 30, 20], "confidence": 0.9}
    ]
    with pytest.raises(ContractViolation):
        validate_grounding_result(payload)


def test_grounding_result_malformed_box_2d_non_numeric():
    payload = _valid_grounding_result()
    payload["bounding_boxes"] = [
        {"label": "building", "box_2d": [10, "x", 30, 40], "confidence": 0.9}
    ]
    with pytest.raises(ContractViolation):
        validate_grounding_result(payload)


def test_grounding_result_bounding_box_extra_key():
    payload = _valid_grounding_result()
    payload["bounding_boxes"] = [
        {"label": "building", "box_2d": [10, 20, 30, 40], "confidence": 0.9, "extra": 1}
    ]
    with pytest.raises(ContractViolation):
        validate_grounding_result(payload)


def test_grounding_result_bounding_box_missing_key():
    payload = _valid_grounding_result()
    payload["bounding_boxes"] = [{"label": "building", "box_2d": [10, 20, 30, 40]}]
    with pytest.raises(ContractViolation):
        validate_grounding_result(payload)


# ---------------------------------------------------------------------------
# validate_change_result
# ---------------------------------------------------------------------------


def test_change_result_valid_passes():
    payload = _valid_change_result()
    assert validate_change_result(payload) == payload


def test_change_result_missing_key():
    payload = _valid_change_result()
    del payload["metrics"]
    with pytest.raises(ContractViolation):
        validate_change_result(payload)


def test_change_result_extra_key():
    payload = _valid_change_result()
    payload["extra"] = "nope"
    with pytest.raises(ContractViolation):
        validate_change_result(payload)


def test_change_result_wrong_type():
    payload = _valid_change_result()
    payload["metrics"] = ["not", "a", "dict"]
    with pytest.raises(ContractViolation):
        validate_change_result(payload)


def test_change_result_confidence_out_of_range():
    payload = _valid_change_result()
    payload["confidence"] = 1.01
    with pytest.raises(ContractViolation):
        validate_change_result(payload)


def test_change_result_bad_confidence_basis():
    payload = _valid_change_result()
    payload["confidence_basis"] = "pixel_math"
    with pytest.raises(ContractViolation):
        validate_change_result(payload)


# ---------------------------------------------------------------------------
# validate_fusion_result
# ---------------------------------------------------------------------------


def test_fusion_result_valid_passes():
    payload = _valid_fusion_result()
    assert validate_fusion_result(payload) == payload


def test_fusion_result_missing_key():
    payload = _valid_fusion_result()
    del payload["evidence"]
    with pytest.raises(ContractViolation):
        validate_fusion_result(payload)


def test_fusion_result_extra_key():
    payload = _valid_fusion_result()
    payload["debug_info"] = {}
    with pytest.raises(ContractViolation):
        validate_fusion_result(payload)


def test_fusion_result_wrong_type():
    payload = _valid_fusion_result()
    payload["text_response"] = 12345
    with pytest.raises(ContractViolation):
        validate_fusion_result(payload)


def test_fusion_result_confidence_out_of_range():
    payload = _valid_fusion_result()
    payload["confidence"] = -1.0
    with pytest.raises(ContractViolation):
        validate_fusion_result(payload)


def test_fusion_result_bad_confidence_basis():
    payload = _valid_fusion_result()
    payload["confidence_basis"] = "trust_me_bro"
    with pytest.raises(ContractViolation):
        validate_fusion_result(payload)


# ---------------------------------------------------------------------------
# validate_execution_trace
# ---------------------------------------------------------------------------


def test_execution_trace_valid_passes():
    payload = _valid_execution_trace()
    assert validate_execution_trace(payload) == payload


def test_execution_trace_missing_key():
    payload = _valid_execution_trace()
    del payload["routing"]
    with pytest.raises(ContractViolation):
        validate_execution_trace(payload)


def test_execution_trace_extra_key():
    payload = _valid_execution_trace()
    payload["not_in_contract"] = 1
    with pytest.raises(ContractViolation):
        validate_execution_trace(payload)


def test_execution_trace_wrong_type():
    payload = _valid_execution_trace()
    payload["inputs"] = "not-a-list"
    with pytest.raises(ContractViolation):
        validate_execution_trace(payload)


def test_execution_trace_task_selected_not_in_enum():
    payload = _valid_execution_trace()
    payload["task_selected"] = "summarize"
    with pytest.raises(ContractViolation):
        validate_execution_trace(payload)


def test_execution_trace_routing_mechanism_not_in_enum():
    payload = _valid_execution_trace()
    payload["routing"] = dict(payload["routing"])
    payload["routing"]["mechanism"] = "llm_vibes"
    with pytest.raises(ContractViolation):
        validate_execution_trace(payload)


def test_execution_trace_input_modality_not_in_enum():
    payload = _valid_execution_trace()
    payload["inputs"] = copy.deepcopy(payload["inputs"])
    payload["inputs"][0]["modality"] = "hyperspectral"
    with pytest.raises(ContractViolation):
        validate_execution_trace(payload)


def test_execution_trace_missing_key_nested_models_used():
    payload = _valid_execution_trace()
    payload["models_used"] = [
        {"role": "vlm", "name": "qwen2-vl", "revision": "v1", "precision": "fp16"}
    ]
    with pytest.raises(ContractViolation):
        validate_execution_trace(payload)


def test_execution_trace_extra_key_nested_artifacts():
    payload = _valid_execution_trace()
    payload["artifacts"] = dict(payload["artifacts"])
    payload["artifacts"]["extra_artifact"] = "path.png"
    with pytest.raises(ContractViolation):
        validate_execution_trace(payload)


# ---------------------------------------------------------------------------
# validate_band_mapping
# ---------------------------------------------------------------------------


def test_band_mapping_valid_passes():
    payload = _valid_band_mapping()
    assert validate_band_mapping(payload) == payload


def test_band_mapping_missing_key():
    payload = _valid_band_mapping()
    del payload["fill_strategy"]
    with pytest.raises(ContractViolation):
        validate_band_mapping(payload)


def test_band_mapping_extra_key():
    payload = _valid_band_mapping()
    payload["extra"] = "nope"
    with pytest.raises(ContractViolation):
        validate_band_mapping(payload)


def test_band_mapping_wrong_type():
    payload = _valid_band_mapping()
    payload["slots_filled"] = "B04,B03,B02"
    with pytest.raises(ContractViolation):
        validate_band_mapping(payload)


def test_band_mapping_accepts_unknown_source_modality():
    # §4.5 (resolved 2026-08-29 via §5.5): "unknown" IS legal here. §4.4's
    # precedence chain can fail to decide on an unrecognised TIFF, and benclip
    # must then fail soft — map by band count and disclose the uncertainty in
    # the trace — rather than refusing to run on the graded data.
    payload = _valid_band_mapping()
    payload["source_modality"] = "unknown"
    assert validate_band_mapping(payload) is not None


def test_band_mapping_bad_source_modality_garbage():
    payload = _valid_band_mapping()
    payload["source_modality"] = "hyperspectral"
    with pytest.raises(ContractViolation):
        validate_band_mapping(payload)


# ---------------------------------------------------------------------------
# Drift guard: the TypedDict annotations and each validator's own required-key
# set are two independent encodings of the same contract (one for static
# checking, one for runtime enforcement). Nothing else ties them together, so
# a future edit to one without the other would pass every test above while
# silently diverging from PLAN.md §4 (exactly what §5.5 exists to prevent).
# These tests fail loudly if that ever happens: the fixture's key set must
# equal the TypedDict's declared keys, and the validator must reject the
# payload with *any single key* removed.
# ---------------------------------------------------------------------------

_DRIFT_GUARD_CASES = [
    (VQAResult, validate_vqa_result, _valid_vqa_result),
    (CaptionResult, validate_caption_result, _valid_caption_result),
    (GroundingResult, validate_grounding_result, _valid_grounding_result),
    (ChangeResult, validate_change_result, _valid_change_result),
    (FusionResult, validate_fusion_result, _valid_fusion_result),
    (ExecutionTrace, validate_execution_trace, _valid_execution_trace),
    (BandMapping, validate_band_mapping, _valid_band_mapping),
]


@pytest.mark.parametrize(
    "typed_dict,validator,make_payload",
    _DRIFT_GUARD_CASES,
    ids=[c[0].__name__ for c in _DRIFT_GUARD_CASES],
)
def test_typeddict_keys_match_validator_required_keys(typed_dict, validator, make_payload):
    payload = make_payload()
    # The fixture (a real, valid payload) must have exactly the keys the
    # TypedDict declares -- neither side has a key the other lacks.
    assert set(payload.keys()) == set(typed_dict.__annotations__.keys())
    # And the validator must actually require every one of those keys:
    # dropping any single one must be rejected.
    for key in list(payload.keys()):
        broken = {k: v for k, v in payload.items() if k != key}
        with pytest.raises(ContractViolation):
            validator(broken)


def test_bounding_box_typed_dict_keys_match_fixture():
    payload = _valid_bounding_box()
    assert set(payload.keys()) == set(BoundingBox.__annotations__.keys())


def test_label_prediction_typed_dict_keys_match_expected_shape():
    # LabelPrediction has no dedicated validator (not required by PLAN.md
    # §4.5 -- only band_mapping is validated directly); this just guards
    # the TypedDict's declared shape against silent drift.
    payload = {"label": "forest", "score": 0.83}
    assert set(payload.keys()) == set(LabelPrediction.__annotations__.keys())
