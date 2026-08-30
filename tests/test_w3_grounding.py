"""
Tests for satquery/specialists/grounding.py (PLAN.md §4.1/§7 W3).

Split per the work order: pure-logic unit tests run everywhere (no GPU, no
weights); the end-to-end test is gated on CUDA + a successful real model
load, so CI-without-GPU still passes the rest of this file.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from satquery.contracts import validate_grounding_result
from satquery.specialists.grounding import (
    _coerce_label,
    _draw_overlay,
    _normalize_query,
    _to_contract_box,
    run_grounding,
)

CUDA_AVAILABLE = torch.cuda.is_available()


# --------------------------------------------------------------------------- #
# Pure-logic tests -- no model, no GPU.
# --------------------------------------------------------------------------- #


def test_normalize_query_lowercases_and_periods():
    assert _normalize_query("A Road") == "a road."


def test_normalize_query_does_not_double_period():
    assert _normalize_query("a road.") == "a road."


def test_normalize_query_empty_falls_back_to_object():
    assert _normalize_query("   ") == "object."


def test_coerce_label_prefers_clean_string():
    assert _coerce_label("a road", "fallback") == "a road"


def test_coerce_label_falls_back_on_none():
    assert _coerce_label(None, "fallback") == "fallback"


def test_coerce_label_falls_back_on_empty_string():
    assert _coerce_label("", "fallback") == "fallback"


def test_coerce_label_falls_back_on_bare_int_label():
    """Some transformers versions return label ids under `labels` instead of
    decoded strings; must not leak a stray int/int-as-string into the payload."""
    assert _coerce_label(3, "fallback") == "fallback"
    assert _coerce_label("7", "fallback") == "fallback"


def test_coerce_label_strips_whitespace():
    assert _coerce_label("  a car  ", "fallback") == "a car"


def test_to_contract_box_converts_xyxy_to_ymin_xmin_ymax_xmax():
    box = _to_contract_box([10.0, 20.0, 110.0, 220.0], "a road", 0.55, "fallback")
    assert box["box_2d"] == [20.0, 10.0, 220.0, 110.0]
    assert box["label"] == "a road"
    assert box["confidence"] == pytest.approx(0.55)


def test_to_contract_box_clamps_inverted_coordinates():
    """Defends the contract's ymin<=ymax / xmin<=xmax invariant even if the
    raw model output is inverted for some reason."""
    box = _to_contract_box([50.0, 5.0, 10.0, 60.0], "x", 0.1, "fallback")
    ymin, xmin, ymax, xmax = box["box_2d"]
    assert ymin <= ymax
    assert xmin <= xmax


def test_to_contract_box_coerces_numpy_scalars_to_python_floats():
    """PLAN.md §4.5: validators (and json.dump) reject numpy scalar types."""
    box = _to_contract_box(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "a road",
        np.float32(0.42),
        "fallback",
    )
    assert all(isinstance(v, float) for v in box["box_2d"])
    assert isinstance(box["confidence"], float)


def test_to_contract_box_none_score_defaults_to_zero():
    box = _to_contract_box([0.0, 0.0, 1.0, 1.0], "x", None, "fallback")
    assert box["confidence"] == 0.0


def test_draw_overlay_writes_a_real_file_with_boxes(tmp_path):
    display_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = [{"label": "a road", "box_2d": [10.0, 5.0, 40.0, 50.0], "confidence": 0.7}]
    out_path = str(tmp_path / "overlay.png")

    result_path = _draw_overlay(display_rgb, boxes, out_path)

    assert os.path.isfile(result_path)
    assert os.path.getsize(result_path) > 0


def test_draw_overlay_writes_a_file_with_zero_boxes():
    """R7 / honesty rule: zero detections still produces a real overlay image,
    never a fabricated box."""
    display_rgb = np.zeros((32, 32, 3), dtype=np.uint8)

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "sub", "overlay.png")
        result_path = _draw_overlay(display_rgb, [], out_path)
        assert os.path.isfile(result_path)


def test_run_grounding_stub_notice_is_gone():
    """Guards against accidentally leaving the W0 stub body in place."""
    import inspect

    from satquery.specialists import grounding as grounding_mod

    src = inspect.getsource(grounding_mod.run_grounding)
    assert "stub" not in src.lower()


# --------------------------------------------------------------------------- #
# End-to-end test -- requires CUDA and downloads/loads the real Grounding DINO
# tiny checkpoint. Skips cleanly when CUDA is unavailable so CI-without-GPU
# still passes the rest of this suite (per the work order).
# --------------------------------------------------------------------------- #

_RSVQA_IMAGE = os.path.join("data", "rsvqa_lr", "Images_LR", "232.tif")


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a CUDA device")
@pytest.mark.skipif(not os.path.isfile(_RSVQA_IMAGE), reason="RSVQA-LR sample image not on disk")
def test_run_grounding_end_to_end_on_real_rsvqa_image():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    result = run_grounding(_RSVQA_IMAGE, "a road")
    validate_grounding_result(result)

    assert isinstance(result["text_response"], str) and result["text_response"]
    assert result["overlay_path"] is not None
    assert os.path.isfile(result["overlay_path"])
    for box in result["bounding_boxes"]:
        ymin, xmin, ymax, xmax = box["box_2d"]
        assert ymin <= ymax
        assert xmin <= xmax
        assert isinstance(box["confidence"], float)

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    # Budget from the work order: ~2.5 GB available to W3 alone (W2 already
    # holds ~1.4 GB elsewhere on the same 3.9 GB card). This process's own
    # peak allocation (torch.cuda.max_memory_allocated) for grounding alone
    # measured ~1.55 GB during development -- see docs/status/W3.md, which
    # also reports the board-wide nvidia-smi figure since max_memory_allocated
    # is per-process and excludes the CUDA context.
    assert peak_mb < 2500, f"grounding peak allocated {peak_mb:.0f} MB exceeds budget"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a CUDA device")
@pytest.mark.skipif(not os.path.isfile(_RSVQA_IMAGE), reason="RSVQA-LR sample image not on disk")
def test_run_grounding_zero_detections_is_honest_not_fabricated():
    """A threshold high enough that nothing clears it must yield an empty
    box list, not a fabricated detection."""
    from satquery.specialists import grounding as grounding_mod

    old_threshold = grounding_mod._DETECTION_THRESHOLD
    grounding_mod._DETECTION_THRESHOLD = 0.999
    try:
        result = run_grounding(_RSVQA_IMAGE, "a road")
        validate_grounding_result(result)
        assert result["bounding_boxes"] == []
        assert result["confidence"] == 0.0
        assert os.path.isfile(result["overlay_path"])
    finally:
        grounding_mod._DETECTION_THRESHOLD = old_threshold
