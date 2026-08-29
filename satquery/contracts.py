"""Machine-readable copy of PLAN.md §4 (Frozen Contracts).

This module is the single source of truth for every payload shape that
crosses a module boundary in this repo: the five specialist function
return values (§4.1), the execution trace written to
``runs/<run_id>/trace.json`` (§4.2), and the `benclip` band-mapping
payload (§4.5).

Everyone imports from here. Nothing in this file may diverge from
PLAN.md §4 without following the §5.5 contract-change protocol (edit
both files in the same commit, flag it to the human). Do not "fix" or
"improve" a contract here — if PLAN.md §4 looks wrong, implement it as
written and report the concern.

Design notes:
- TypedDicts, not dataclasses, for the JSON-shaped payloads: they map
  directly onto the dict literals every specialist actually returns and
  onto ``json.dump``/``json.load`` without a conversion step.
- Every payload has a ``validate_*`` function. Each takes the payload,
  and either returns it unchanged (so call sites can write
  ``return validate_vqa_result(payload)``) or raises
  ``ContractViolation`` naming the offending key.
- Validators are strict: they reject missing required keys, unexpected
  extra keys, wrong types, and out-of-domain enum values. A permissive
  validator that rubber-stamps a subtly-wrong payload is worse than
  useless here — the whole point is to catch a drifting caller before
  it reaches ``runs/*/trace.json`` or the app.
- Pure stdlib + ``typing``. No pydantic — it is not a project
  dependency (§5.6/pyproject.toml is W0-owned and does not list it).
- Validators use exact ``isinstance`` checks (``float``/``int``/``str``,
  not numpy scalar types). A ``numpy.float32`` confidence or
  ``numpy.int64`` band count will be REJECTED. This is deliberate, not
  an oversight: a trace payload has to survive ``json.dump`` too, and
  ``json`` chokes on numpy scalars the same way. Callers (specialists,
  benclip, the controller) must coerce with ``float(...)`` / ``int(...)``
  before building a return payload or a trace entry.
"""

from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

__all__ = [
    # exceptions
    "ContractViolation",
    # enums / literals
    "ConfidenceBasis",
    "TaskName",
    "Modality",
    "CONFIDENCE_BASES",
    "TASK_NAMES",
    "MODALITIES",
    "BAND_MAPPING_MODALITIES",
    "ROUTING_MECHANISMS",
    # §4.1 payload shapes
    "VQAResult",
    "CaptionResult",
    "BoundingBox",
    "GroundingResult",
    "ChangeResult",
    "FusionResult",
    # §4.2 execution trace shapes
    "RoutingInfo",
    "InputInfo",
    "ValidationInfo",
    "ModelUsed",
    "ArtifactsInfo",
    "TimingsMs",
    "ExecutionTrace",
    # §4.5 benclip band-mapping shapes
    "BandMapping",
    "LabelPrediction",
    # validators
    "validate_vqa_result",
    "validate_caption_result",
    "validate_grounding_result",
    "validate_change_result",
    "validate_fusion_result",
    "validate_execution_trace",
    "validate_band_mapping",
]


class ContractViolation(Exception):
    """Raised by a ``validate_*`` function when a payload does not match
    the frozen contract in PLAN.md §4. The message names the offending
    key so a failing caller is quick to diagnose."""


# ---------------------------------------------------------------------------
# §4.1 / §4.2 / §4.4 enums, encoded as Literal types (for static checking)
# plus frozensets of the same values (for runtime membership checks).
# ---------------------------------------------------------------------------

ConfidenceBasis = Literal["stub", "heuristic", "calibrated", "model_logprob"]
CONFIDENCE_BASES: frozenset = frozenset({"stub", "heuristic", "calibrated", "model_logprob"})

TaskName = Literal["vqa", "caption", "grounding", "change", "fusion"]
TASK_NAMES: frozenset = frozenset({"vqa", "caption", "grounding", "change", "fusion"})

# §4.4: modality tagging on an ingested raster. Includes "unknown" for the
# case where no mechanism could confidently decide.
Modality = Literal["optical", "msi", "sar", "unknown"]
MODALITIES: frozenset = frozenset({"optical", "msi", "sar", "unknown"})

# §4.5: predict_labels' band_mapping.source_modality accepts the full Modality
# set INCLUDING "unknown". §4.4's precedence chain can legitimately fail to
# decide, and an unrecognised TIFF on the hidden ISRO eval set is exactly when
# that happens. benclip must then fail soft — map by band count alone and report
# "unknown" so the trace discloses the uncertainty — rather than refusing.
# A degraded, labelled-uncertain answer scores; a hard failure scores zero.
# (Resolved 2026-08-29 via PLAN.md §5.5; §4.5 originally omitted "unknown".)
BAND_MAPPING_MODALITIES: frozenset = frozenset(MODALITIES)

# §3.4 / §4.2 routing.mechanism
ROUTING_MECHANISMS: frozenset = frozenset({"rule", "exemplar_nn"})


# ---------------------------------------------------------------------------
# §4.1 — Specialist functions
# ---------------------------------------------------------------------------


class VQAResult(TypedDict):
    text_response: str
    confidence: float
    confidence_basis: ConfidenceBasis
    evidence: dict


class CaptionResult(TypedDict):
    text_response: str
    confidence: float
    confidence_basis: ConfidenceBasis
    evidence: dict


class BoundingBox(TypedDict):
    label: str
    box_2d: List[float]  # [ymin, xmin, ymax, xmax], pixel coords
    confidence: float


class GroundingResult(TypedDict):
    text_response: str
    bounding_boxes: List[BoundingBox]
    overlay_path: Optional[str]
    confidence: float
    confidence_basis: ConfidenceBasis


class ChangeResult(TypedDict):
    text_response: str
    change_mask_path: Optional[str]
    overlay_path: Optional[str]
    metrics: dict
    confidence: float
    confidence_basis: ConfidenceBasis
    evidence: dict


class FusionResult(TypedDict):
    text_response: str
    agreement_map_path: Optional[str]
    overlay_path: Optional[str]
    evidence: dict
    confidence: float
    confidence_basis: ConfidenceBasis


# ---------------------------------------------------------------------------
# §4.2 — Execution trace
# ---------------------------------------------------------------------------


class RoutingInfo(TypedDict):
    mechanism: str  # "rule" | "exemplar_nn"
    matched: str
    score: float
    alternatives_considered: List[str]


class InputInfo(TypedDict):
    path: str
    modality: Modality
    bands: int
    shape: List[int]  # [height, width]
    format: str
    crs: Optional[str]
    checks_passed: bool


class ValidationInfo(TypedDict):
    passed: bool
    warnings: List[str]
    errors: List[str]


class ModelUsed(TypedDict):
    role: str
    name: str
    revision: str
    precision: str
    adapter: Optional[str]
    # `device` added 2026-08-29 via PLAN.md §5.5: modelpool.get_execution_metadata()
    # reports it, and an audit trace that omits whether a run was on GPU or CPU is
    # less reproducible. PLAN.md §4.2 updated in the same change.
    device: str


class ArtifactsInfo(TypedDict):
    mask: Optional[str]
    overlay: Optional[str]
    report: Optional[str]


class TimingsMs(TypedDict):
    routing: int
    validation: int
    inference: int
    total: int


class ExecutionTrace(TypedDict):
    run_id: str
    timestamp: str
    query: str
    task_selected: TaskName
    routing: RoutingInfo
    inputs: List[InputInfo]
    validation: ValidationInfo
    models_used: List[ModelUsed]
    parameters: dict
    result: dict
    artifacts: ArtifactsInfo
    timings_ms: TimingsMs


# ---------------------------------------------------------------------------
# §4.5 — benclip band-mapping contract
# ---------------------------------------------------------------------------


class BandMapping(TypedDict):
    slots_filled: List[str]
    slots_absent: List[str]
    fill_strategy: str
    source_modality: str  # "optical"|"msi"|"sar"|"unknown" — see BAND_MAPPING_MODALITIES


class LabelPrediction(TypedDict):
    label: str
    score: float


# ---------------------------------------------------------------------------
# Internal validation helpers. Not part of the public contract surface, but
# every one of them is exercised indirectly by every test in
# tests/test_w0_contracts.py, so none of this is dead code (§5.4).
# ---------------------------------------------------------------------------


def _fail(message: str) -> None:
    raise ContractViolation(message)


def _ensure_mapping(payload: object, context: str) -> None:
    if not isinstance(payload, dict):
        _fail(f"{context}: expected a dict, got {type(payload).__name__}")


def _check_required_keys(payload: dict, required: set, context: str) -> None:
    missing = required - set(payload.keys())
    if missing:
        _fail(f"{context}: missing required key(s): {sorted(missing)}")


def _check_no_extra_keys(payload: dict, allowed: set, context: str) -> None:
    extra = set(payload.keys()) - allowed
    if extra:
        _fail(f"{context}: unexpected key(s): {sorted(extra)}")


def _check_str(value: object, key: str, context: str) -> None:
    if not isinstance(value, str):
        _fail(f"{context}: key '{key}' must be a str, got {type(value).__name__}")


def _check_optional_str(value: object, key: str, context: str) -> None:
    if value is not None and not isinstance(value, str):
        _fail(f"{context}: key '{key}' must be a str or None, got {type(value).__name__}")


def _check_bool(value: object, key: str, context: str) -> None:
    if not isinstance(value, bool):
        _fail(f"{context}: key '{key}' must be a bool, got {type(value).__name__}")


def _check_number(value: object, key: str, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{context}: key '{key}' must be a number, got {type(value).__name__}")


def _check_int(value: object, key: str, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{context}: key '{key}' must be an int, got {type(value).__name__}")


def _check_dict(value: object, key: str, context: str) -> None:
    if not isinstance(value, dict):
        _fail(f"{context}: key '{key}' must be a dict, got {type(value).__name__}")


def _check_list(value: object, key: str, context: str) -> None:
    if not isinstance(value, list):
        _fail(f"{context}: key '{key}' must be a list, got {type(value).__name__}")


def _check_list_of_str(value: object, key: str, context: str) -> None:
    _check_list(value, key, context)
    for i, item in enumerate(value):  # type: ignore[union-attr]
        if not isinstance(item, str):
            _fail(f"{context}: key '{key}[{i}]' must be a str, got {type(item).__name__}")


def _check_confidence(value: object, key: str, context: str) -> None:
    _check_number(value, key, context)
    if not (0.0 <= float(value) <= 1.0):  # type: ignore[arg-type]
        _fail(f"{context}: key '{key}' must be within [0,1], got {value!r}")


def _check_confidence_basis(value: object, key: str, context: str) -> None:
    _check_str(value, key, context)
    if value not in CONFIDENCE_BASES:
        _fail(
            f"{context}: key '{key}' must be one of {sorted(CONFIDENCE_BASES)}, "
            f"got {value!r}"
        )


def _validate_bounding_box(box: object, index: int, context: str) -> None:
    ctx = f"{context}.bounding_boxes[{index}]"
    _ensure_mapping(box, ctx)
    assert isinstance(box, dict)
    required = {"label", "box_2d", "confidence"}
    _check_required_keys(box, required, ctx)
    _check_no_extra_keys(box, required, ctx)
    _check_str(box["label"], "label", ctx)
    _check_confidence(box["confidence"], "confidence", ctx)

    box_2d = box["box_2d"]
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
        _fail(
            f"{ctx}: key 'box_2d' must be a list of exactly 4 numbers "
            f"[ymin, xmin, ymax, xmax], got {box_2d!r}"
        )
    for i, v in enumerate(box_2d):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            _fail(f"{ctx}: key 'box_2d[{i}]' must be numeric, got {type(v).__name__}")
    ymin, xmin, ymax, xmax = box_2d
    if ymin > ymax:
        _fail(f"{ctx}: key 'box_2d' invalid: ymin ({ymin}) > ymax ({ymax})")
    if xmin > xmax:
        _fail(f"{ctx}: key 'box_2d' invalid: xmin ({xmin}) > xmax ({xmax})")


def _validate_text_result(payload: object, context: str) -> dict:
    """Shared shape for run_vqa/run_caption: text_response, confidence,
    confidence_basis, evidence — identical per §4.1."""
    _ensure_mapping(payload, context)
    assert isinstance(payload, dict)
    required = {"text_response", "confidence", "confidence_basis", "evidence"}
    _check_required_keys(payload, required, context)
    _check_no_extra_keys(payload, required, context)
    _check_str(payload["text_response"], "text_response", context)
    _check_confidence(payload["confidence"], "confidence", context)
    _check_confidence_basis(payload["confidence_basis"], "confidence_basis", context)
    _check_dict(payload["evidence"], "evidence", context)
    return payload


# ---------------------------------------------------------------------------
# Public validators — one per §4 contract.
# ---------------------------------------------------------------------------


def validate_vqa_result(payload: object) -> VQAResult:
    """Validate a run_vqa return payload against §4.1. Returns the payload
    unchanged on success; raises ContractViolation on any mismatch."""
    return _validate_text_result(payload, "vqa_result")  # type: ignore[return-value]


def validate_caption_result(payload: object) -> CaptionResult:
    """Validate a run_caption return payload against §4.1. Same shape as
    run_vqa's result."""
    return _validate_text_result(payload, "caption_result")  # type: ignore[return-value]


def validate_grounding_result(payload: object) -> GroundingResult:
    """Validate a run_grounding return payload against §4.1, including
    each bounding box's ``box_2d`` ordering/consistency."""
    context = "grounding_result"
    _ensure_mapping(payload, context)
    assert isinstance(payload, dict)
    required = {"text_response", "bounding_boxes", "overlay_path", "confidence", "confidence_basis"}
    _check_required_keys(payload, required, context)
    _check_no_extra_keys(payload, required, context)

    _check_str(payload["text_response"], "text_response", context)

    boxes = payload["bounding_boxes"]
    _check_list(boxes, "bounding_boxes", context)
    for i, box in enumerate(boxes):  # type: ignore[arg-type]
        _validate_bounding_box(box, i, context)

    _check_optional_str(payload["overlay_path"], "overlay_path", context)
    _check_confidence(payload["confidence"], "confidence", context)
    _check_confidence_basis(payload["confidence_basis"], "confidence_basis", context)
    return payload  # type: ignore[return-value]


def validate_change_result(payload: object) -> ChangeResult:
    """Validate a run_change return payload against §4.1."""
    context = "change_result"
    _ensure_mapping(payload, context)
    assert isinstance(payload, dict)
    required = {
        "text_response",
        "change_mask_path",
        "overlay_path",
        "metrics",
        "confidence",
        "confidence_basis",
        "evidence",
    }
    _check_required_keys(payload, required, context)
    _check_no_extra_keys(payload, required, context)

    _check_str(payload["text_response"], "text_response", context)
    _check_optional_str(payload["change_mask_path"], "change_mask_path", context)
    _check_optional_str(payload["overlay_path"], "overlay_path", context)
    _check_dict(payload["metrics"], "metrics", context)
    _check_confidence(payload["confidence"], "confidence", context)
    _check_confidence_basis(payload["confidence_basis"], "confidence_basis", context)
    _check_dict(payload["evidence"], "evidence", context)
    return payload  # type: ignore[return-value]


def validate_fusion_result(payload: object) -> FusionResult:
    """Validate a run_fusion return payload against §4.1."""
    context = "fusion_result"
    _ensure_mapping(payload, context)
    assert isinstance(payload, dict)
    required = {
        "text_response",
        "agreement_map_path",
        "overlay_path",
        "evidence",
        "confidence",
        "confidence_basis",
    }
    _check_required_keys(payload, required, context)
    _check_no_extra_keys(payload, required, context)

    _check_str(payload["text_response"], "text_response", context)
    _check_optional_str(payload["agreement_map_path"], "agreement_map_path", context)
    _check_optional_str(payload["overlay_path"], "overlay_path", context)
    _check_dict(payload["evidence"], "evidence", context)
    _check_confidence(payload["confidence"], "confidence", context)
    _check_confidence_basis(payload["confidence_basis"], "confidence_basis", context)
    return payload  # type: ignore[return-value]


def _validate_routing(routing: object, context: str) -> None:
    ctx = f"{context}.routing"
    _ensure_mapping(routing, ctx)
    assert isinstance(routing, dict)
    required = {"mechanism", "matched", "score", "alternatives_considered"}
    _check_required_keys(routing, required, ctx)
    _check_no_extra_keys(routing, required, ctx)

    _check_str(routing["mechanism"], "mechanism", ctx)
    if routing["mechanism"] not in ROUTING_MECHANISMS:
        _fail(
            f"{ctx}: key 'mechanism' must be one of {sorted(ROUTING_MECHANISMS)}, "
            f"got {routing['mechanism']!r}"
        )
    _check_str(routing["matched"], "matched", ctx)
    _check_number(routing["score"], "score", ctx)
    _check_list_of_str(routing["alternatives_considered"], "alternatives_considered", ctx)


def _validate_input_info(item: object, index: int, context: str) -> None:
    ctx = f"{context}.inputs[{index}]"
    _ensure_mapping(item, ctx)
    assert isinstance(item, dict)
    required = {"path", "modality", "bands", "shape", "format", "crs", "checks_passed"}
    _check_required_keys(item, required, ctx)
    _check_no_extra_keys(item, required, ctx)

    _check_str(item["path"], "path", ctx)
    _check_str(item["modality"], "modality", ctx)
    if item["modality"] not in MODALITIES:
        _fail(f"{ctx}: key 'modality' must be one of {sorted(MODALITIES)}, got {item['modality']!r}")
    _check_int(item["bands"], "bands", ctx)

    shape = item["shape"]
    _check_list(shape, "shape", ctx)
    if len(shape) != 2:  # type: ignore[arg-type]
        _fail(f"{ctx}: key 'shape' must have exactly 2 elements [height, width], got {shape!r}")
    for i, v in enumerate(shape):  # type: ignore[arg-type]
        if isinstance(v, bool) or not isinstance(v, int):
            _fail(f"{ctx}: key 'shape[{i}]' must be an int, got {type(v).__name__}")

    _check_str(item["format"], "format", ctx)
    _check_optional_str(item["crs"], "crs", ctx)
    _check_bool(item["checks_passed"], "checks_passed", ctx)


def _validate_validation_info(validation: object, context: str) -> None:
    ctx = f"{context}.validation"
    _ensure_mapping(validation, ctx)
    assert isinstance(validation, dict)
    required = {"passed", "warnings", "errors"}
    _check_required_keys(validation, required, ctx)
    _check_no_extra_keys(validation, required, ctx)

    _check_bool(validation["passed"], "passed", ctx)
    _check_list_of_str(validation["warnings"], "warnings", ctx)
    _check_list_of_str(validation["errors"], "errors", ctx)


def _validate_model_used(item: object, index: int, context: str) -> None:
    ctx = f"{context}.models_used[{index}]"
    _ensure_mapping(item, ctx)
    assert isinstance(item, dict)
    required = {"role", "name", "revision", "precision", "adapter", "device"}
    _check_required_keys(item, required, ctx)
    _check_no_extra_keys(item, required, ctx)

    _check_str(item["role"], "role", ctx)
    _check_str(item["name"], "name", ctx)
    _check_str(item["revision"], "revision", ctx)
    _check_str(item["precision"], "precision", ctx)
    _check_optional_str(item["adapter"], "adapter", ctx)
    _check_str(item["device"], "device", ctx)


def _validate_artifacts(artifacts: object, context: str) -> None:
    ctx = f"{context}.artifacts"
    _ensure_mapping(artifacts, ctx)
    assert isinstance(artifacts, dict)
    required = {"mask", "overlay", "report"}
    _check_required_keys(artifacts, required, ctx)
    _check_no_extra_keys(artifacts, required, ctx)

    _check_optional_str(artifacts["mask"], "mask", ctx)
    _check_optional_str(artifacts["overlay"], "overlay", ctx)
    _check_optional_str(artifacts["report"], "report", ctx)


def _validate_timings(timings: object, context: str) -> None:
    ctx = f"{context}.timings_ms"
    _ensure_mapping(timings, ctx)
    assert isinstance(timings, dict)
    required = {"routing", "validation", "inference", "total"}
    _check_required_keys(timings, required, ctx)
    _check_no_extra_keys(timings, required, ctx)

    for key in required:
        _check_number(timings[key], key, ctx)


def validate_execution_trace(payload: object) -> ExecutionTrace:
    """Validate a §4.2 execution-trace payload (the object written to
    ``runs/<run_id>/trace.json``), including all nested shapes."""
    context = "execution_trace"
    _ensure_mapping(payload, context)
    assert isinstance(payload, dict)
    required = {
        "run_id",
        "timestamp",
        "query",
        "task_selected",
        "routing",
        "inputs",
        "validation",
        "models_used",
        "parameters",
        "result",
        "artifacts",
        "timings_ms",
    }
    _check_required_keys(payload, required, context)
    _check_no_extra_keys(payload, required, context)

    _check_str(payload["run_id"], "run_id", context)
    _check_str(payload["timestamp"], "timestamp", context)
    _check_str(payload["query"], "query", context)

    _check_str(payload["task_selected"], "task_selected", context)
    if payload["task_selected"] not in TASK_NAMES:
        _fail(
            f"{context}: key 'task_selected' must be one of {sorted(TASK_NAMES)}, "
            f"got {payload['task_selected']!r}"
        )

    _validate_routing(payload["routing"], context)

    inputs = payload["inputs"]
    _check_list(inputs, "inputs", context)
    for i, item in enumerate(inputs):  # type: ignore[arg-type]
        _validate_input_info(item, i, context)

    _validate_validation_info(payload["validation"], context)

    models_used = payload["models_used"]
    _check_list(models_used, "models_used", context)
    for i, item in enumerate(models_used):  # type: ignore[arg-type]
        _validate_model_used(item, i, context)

    _check_dict(payload["parameters"], "parameters", context)
    _check_dict(payload["result"], "result", context)

    _validate_artifacts(payload["artifacts"], context)
    _validate_timings(payload["timings_ms"], context)

    return payload  # type: ignore[return-value]


def validate_band_mapping(payload: object) -> BandMapping:
    """Validate the ``band_mapping`` sub-payload returned by
    ``benclip.predict_labels`` per §4.5."""
    context = "band_mapping"
    _ensure_mapping(payload, context)
    assert isinstance(payload, dict)
    required = {"slots_filled", "slots_absent", "fill_strategy", "source_modality"}
    _check_required_keys(payload, required, context)
    _check_no_extra_keys(payload, required, context)

    _check_list_of_str(payload["slots_filled"], "slots_filled", context)
    _check_list_of_str(payload["slots_absent"], "slots_absent", context)
    _check_str(payload["fill_strategy"], "fill_strategy", context)

    source_modality = payload["source_modality"]
    _check_str(source_modality, "source_modality", context)
    if source_modality not in BAND_MAPPING_MODALITIES:
        _fail(
            f"{context}: key 'source_modality' must be one of "
            f"{sorted(BAND_MAPPING_MODALITIES)}, got {source_modality!r}"
        )

    return payload  # type: ignore[return-value]
