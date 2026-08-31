"""Execution trace writer and run orchestrator (PLAN.md §4.2, W6).

This is the audit surface (R5): every run — successful, validation-failed, or
crashed mid-inference — writes ``runs/<run_id>/trace.json`` matching
``contracts.ExecutionTrace``. A crash with a trace explaining what was
attempted is worth far more than a silent crash, so dispatch is wrapped and an
exception still produces a trace before propagating to the caller.

Why ``trace.py`` holds the orchestration entry point: W6's controller owns the
entire run lifecycle (route -> validate -> dispatch -> trace), and per
PLAN.md §6 the only files W6 may create are ``router.py``, ``validate.py`` and
``trace.py``. The trace is not a side effect of a run — it is the deliverable,
so the top-level ``run_query`` that wraps dispatch lives here.

Model bookkeeping (PLAN.md §4.3):
  * The controller never loads or caches a model in a module global.
  * ``models_used`` is built per task as: the task's planned roles from
    ``DEFAULT_REGISTRY`` (which the PS requires be *selectable* and
    *sequenced*), merged with whatever ``model_pool.get_execution_metadata()``
    actually reports once the real W3/W4/W5 specialists load weights. During
    the stub era the planned entries carry the pool's target device and the
    ``models_derivation`` parameter records that they are the plan, not an
    observed load — ever the honesty rule (§5.9).

Numeric coercion (§4.5): every number entering the trace is coerced with
int()/float(), and ``_json_safe`` recursively rewrites numpy scalars to Python
scalars so the payload survives both the strict validators and ``json.dump``.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from satquery.contracts import (
    ArtifactsInfo,
    ExecutionTrace,
    TimingsMs,
    validate_caption_result,
    validate_change_result,
    validate_execution_trace,
    validate_fusion_result,
    validate_grounding_result,
    validate_vqa_result,
)
from satquery.controller.router import RouteResult, extract_grounding_target, route
from satquery.controller.validate import ValidationResult, validate_inputs
from satquery.runtime.modelpool import DEFAULT_REGISTRY, ModelPool, model_pool

from satquery.specialists import change as change_mod
from satquery.specialists import fusion as fusion_mod
from satquery.specialists import grounding as grounding_mod
from satquery.specialists import vqa as vqa_mod

__all__ = [
    "TASK_ROLES",
    "new_run_id",
    "write_trace",
    "run_query",
    "_build_trace",
    "_merge_models",
    "_json_safe",
]

DEFAULT_RUNS_ROOT = "runs"

# Which roles a task selects and sequences (PLAN.md §3.2/§3.3). change and
# fusion must demonstrate >=2 models/tools from the trace: benclip (RS-adapted
# label evidence) + the VLM that verbalises is the documented §3.3 path.
TASK_ROLES: Dict[str, Tuple[str, ...]] = {
    "vqa": ("vqa",),
    "caption": ("vqa",),
    "grounding": ("grounding",),
    "change": ("benclip", "vqa"),
    "fusion": ("benclip", "vqa"),
}

_RESULT_VALIDATORS = {
    "vqa": validate_vqa_result,
    "caption": validate_caption_result,
    "grounding": validate_grounding_result,
    "change": validate_change_result,
    "fusion": validate_fusion_result,
}

_EMPTY_ARTIFACTS: ArtifactsInfo = {"mask": None, "overlay": None, "report": None}


# ---------------------------------------------------------------------------
# Trace-format helpers.
# ---------------------------------------------------------------------------


def new_run_id() -> str:
    return f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    """Recursively coerce numpy scalars/arrays and anything else json chokes on
    into pure-Python values before it enters a trace payload."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    return str(value)


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000.0))


def _target_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _planned_model_entry(role: str) -> Dict[str, Any]:
    spec = DEFAULT_REGISTRY[role]
    return {
        "role": role,
        "name": spec.model_id,
        "revision": spec.revision,
        "precision": spec.precision,
        "adapter": None,
        "device": _target_device(),
    }


def _merge_models(task: str, pool: Optional[ModelPool] = None) -> List[Dict[str, Any]]:
    """Task's planned roles merged with actual load metadata from the pool.

    Actual pool entries (real model_id/device/revision) overwrite planned ones
    role-for-role; planned entries fill any role the pool has not loaded yet
    (the stub era). Only roles belonging to this task's plan are reported, so a
    stale model from an unrelated earlier run never leaks into this trace.
    """
    pool = pool if pool is not None else model_pool
    actuals = {entry["role"]: entry for entry in pool.get_execution_metadata()}
    merged: List[Dict[str, Any]] = []
    for role in TASK_ROLES.get(task, ()):
        if role in actuals:
            merged.append(dict(actuals[role]))
        elif role in DEFAULT_REGISTRY:
            merged.append(_planned_model_entry(role))
    return merged


def _artifacts_from_result(task: str, result: dict) -> ArtifactsInfo:
    if task == "change":
        return {
            "mask": result.get("change_mask_path"),
            "overlay": result.get("overlay_path"),
            "report": None,
        }
    if task == "fusion":
        return {
            "mask": result.get("agreement_map_path"),
            "overlay": result.get("overlay_path"),
            "report": None,
        }
    if task == "grounding":
        return {"mask": None, "overlay": result.get("overlay_path"), "report": None}
    return dict(_EMPTY_ARTIFACTS)


def _timings(
    routing_ms: int, validation_ms: int, inference_ms: int, total_ms: int
) -> TimingsMs:
    return {
        "routing": int(routing_ms),
        "validation": int(validation_ms),
        "inference": int(inference_ms),
        "total": int(total_ms),
    }


def _build_trace(
    *,
    run_id: str,
    query: str,
    decision: RouteResult,
    inputs: List[dict],
    validation: dict,
    models_used: List[dict],
    parameters: dict,
    result: dict,
    artifacts: ArtifactsInfo,
    timings: TimingsMs,
) -> ExecutionTrace:
    trace: ExecutionTrace = {
        "run_id": run_id,
        "timestamp": _now_iso(),
        "query": query,
        "task_selected": decision.task,
        "routing": dict(decision.routing),
        "inputs": [_json_safe(i) for i in inputs],
        "validation": dict(validation),
        "models_used": [_json_safe(m) for m in models_used],
        "parameters": _json_safe(parameters),
        "result": _json_safe(result),
        "artifacts": dict(artifacts),
        "timings_ms": dict(timings),
    }
    validate_execution_trace(trace)
    return trace


def write_trace(trace: ExecutionTrace, runs_root: str = DEFAULT_RUNS_ROOT) -> str:
    """Validate ``trace`` against the §4.2 contract and write it to
    ``runs_root/<run_id>/trace.json``. Returns the written path."""
    run_dir = os.path.join(runs_root, trace["run_id"])
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "trace.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(trace, handle, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# Dispatch.
# ---------------------------------------------------------------------------


def _split_optical_sar(input_infos: Sequence[dict]) -> Tuple[int, int, str]:
    """(optical_index, sar_index, note) for run_fusion's two positional args.

    The first confidently-SAR input becomes the SAR channel; the other input
    becomes the optical channel. If no input is confidently SAR (should only
    happen after a validation warning, never an error), order of upload wins
    and the note says so."""
    sar_idx = next(
        (idx for idx, info in enumerate(input_infos) if info["modality"] == "sar"),
        None,
    )
    if sar_idx is None:
        return 0, 1, "no input confidently tagged SAR; using upload order (optical=1st, SAR=2nd)"
    other = 1 - sar_idx
    if input_infos[other]["modality"] == "sar":
        return other, sar_idx, "both inputs tagged SAR; treating 2nd as the optical channel by upload order"
    opt_label = input_infos[other]["modality"]
    return other, sar_idx, f"optical channel = input {other + 1} ({opt_label}); SAR channel = input {sar_idx + 1}"


def _dispatch(
    task: str,
    paths: Sequence[str],
    input_infos: Sequence[dict],
    query: str,
    selected_image_index: int,
) -> Tuple[dict, Dict[str, Any]]:
    """Run the routed specialist. Returns (result_payload, extra_parameters).
    All five specialists are reachable here — including run_grounding, which
    was dead code in the old draft (§2.5/R7)."""
    if task == "vqa":
        return dict(vqa_mod.run_vqa(paths[selected_image_index], query)), {}
    if task == "caption":
        return dict(vqa_mod.run_caption(paths[selected_image_index])), {}
    if task == "grounding":
        target = extract_grounding_target(query)
        params = {"grounding_target": target}
        return dict(grounding_mod.run_grounding(paths[selected_image_index], target)), params
    if task == "change":
        return dict(change_mod.run_change(paths[0], paths[1], query)), {}
    if task == "fusion":
        opt_idx, sar_idx, note = _split_optical_sar(input_infos)
        params = {"fusion_input_assignment": note}
        return (
            dict(fusion_mod.run_fusion(paths[opt_idx], paths[sar_idx], query)),
            params,
        )
    raise ValueError(f"no dispatcher for task {task!r}")


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def run_query(
    query: str,
    image_paths: Sequence[str],
    *,
    forced_modalities: Optional[Sequence[Optional[str]]] = None,
    runs_root: str = DEFAULT_RUNS_ROOT,
    pool: Optional[ModelPool] = None,
) -> ExecutionTrace:
    """Full agentic round trip: route -> validate -> dispatch -> trace.

    Return value is the validated ``ExecutionTrace`` that was also written to
    ``runs_root/<run_id>/trace.json``. A validation failure still writes and
    returns a trace (with ``validation.passed == False``). A crash mid-
    inference writes a trace describing the attempt and then RE-RAISES the
    original exception.

    ``pool`` defaults to the module-level ``model_pool`` singleton; tests pass
    a fresh ModelPool with injected fake roles.
    """
    pool = pool if pool is not None else model_pool
    t_total_start = monotonic()

    run_id = new_run_id()
    paths = list(image_paths)
    n_inputs = len(paths)

    # --- routing -----------------------------------------------------------
    t_r0 = monotonic()
    decision = route(query, n_inputs=n_inputs)
    t_r1 = monotonic()
    routing_ms = _ms(t_r1 - t_r0)

    # --- validation --------------------------------------------------------
    t_v0 = monotonic()
    vr: ValidationResult = validate_inputs(
        paths,
        decision.task,
        forced_modalities=forced_modalities,
        query=query,
        selected_image_index=decision.selected_image_index,
    )
    t_v1 = monotonic()
    validation_ms = _ms(t_v1 - t_v0)

    validation = vr.to_validation_info()
    parameters: Dict[str, Any] = {
        "selected_image_index": decision.selected_image_index,
        "image_reference": decision.image_reference,
        "input_count": n_inputs,
        "routing_resolution": decision.resolution_note,
        "models_derivation": (
            "task-planned roles from DEFAULT_REGISTRY merged with "
            "model_pool.get_execution_metadata(); a planned entry names a role "
            "the task sequences, and carries observed load metadata once that "
            "role has actually been acquired"
        ),
    }

    if not vr.passed:
        # Refuse dispatch, but still audit the attempt (R5: every run traces).
        trace = _build_trace(
            run_id=run_id,
            query=query,
            decision=decision,
            inputs=vr.input_infos,
            validation=validation,
            models_used=_merge_models(decision.task, pool),
            parameters=parameters,
            result={"status": "validation_failed", "errors": list(vr.errors)},
            artifacts=_EMPTY_ARTIFACTS,
            timings=_timings(routing_ms, validation_ms, 0, _ms(monotonic() - t_total_start)),
        )
        write_trace(trace, runs_root)
        return trace

    # --- dispatch (crashes still trace, then propagate) --------------------
    models_used = _merge_models(decision.task, pool)
    t_i0 = monotonic()
    try:
        result, extra_params = _dispatch(
            decision.task, paths, vr.input_infos, query, decision.selected_image_index
        )
        _RESULT_VALIDATORS[decision.task](result)
    except Exception as exc:
        t_i1 = monotonic()
        crash_result: Dict[str, Any] = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        trace = _build_trace(
            run_id=run_id,
            query=query,
            decision=decision,
            inputs=vr.input_infos,
            validation=validation,
            models_used=models_used,
            parameters=parameters,
            result=crash_result,
            artifacts=_EMPTY_ARTIFACTS,
            timings=_timings(
                routing_ms, validation_ms, _ms(t_i1 - t_i0), _ms(monotonic() - t_total_start)
            ),
        )
        write_trace(trace, runs_root)
        raise

    t_i1 = monotonic()
    inference_ms = _ms(t_i1 - t_i0)

    parameters.update(extra_params)
    parameters["result_validated"] = True

    trace = _build_trace(
        run_id=run_id,
        query=query,
        decision=decision,
        inputs=vr.input_infos,
        validation=validation,
        models_used=models_used,
        parameters=parameters,
        result=_json_safe(result),
        artifacts=_artifacts_from_result(decision.task, result),
        timings=_timings(routing_ms, validation_ms, inference_ms, _ms(monotonic() - t_total_start)),
    )
    write_trace(trace, runs_root)
    return trace