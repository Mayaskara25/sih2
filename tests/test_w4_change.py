"""W4 tests — Bi-temporal change analysis specialist (PLAN.md §7 W4).

SCOPE NOTE: PLAN.md §7's W4 acceptance test names CDVQA (>=100 questions) and
SECOND (>=20 reference masks). Neither is obtainable — both are Google
Drive/Baidu hosted with no direct HTTP link, and W1 could not fetch them
(docs/status/W1.md). This suite exercises PLAN.md §8's documented fallback
line instead: classical CV + benclip labels, no trained change head. See
docs/status/W4.md for the full, deliberate accounting of that substitution.

FIXTURE HONESTY (PLAN.md W4 brief, "no true bi-temporal pair exists locally"):
  - "real scenes" tests use two DIFFERENT real BigEarthNet S2 patches. These
    are never described as a genuine same-location bi-temporal pair anywhere
    (in this file, in change.py, or in docs/status/W4.md) — they exercise the
    full mechanics (registration decision, differencing, benclip on both
    dates) on real imagery, nothing more.
  - "synthetic known-change" tests start from ONE real BigEarthNet patch and
    apply a KNOWN, documented modification (a fixed-location intensity bump
    over a fixed bounding box), then assert the mask recovers that exact
    region. This is the correctness signal: we know the ground truth because
    we made it.
  - "featureless" tests are fully synthetic (uniform + low-amplitude noise,
    no georeferencing) — measured directly (see docs/status/W4.md) to leave
    ORB with zero keypoints on both frames.

THE TEST THAT MATTERS MOST: `test_swap_t1_changes_result_on_derived_quantities_only`.
Learn from W5's mistake (PLAN.md W4 brief): its equivalent test initially
accepted an echoed, unequal input path as proof of reading the second image,
which is trivially true for any two distinct paths. This test instead: (a)
requires benclip to have actually produced a non-empty per-class delta on
BOTH swapped calls (otherwise comparing two empty lists would be vacuous),
(b) asserts the t0-side benclip labels are IDENTICAL across both calls (same
file, deterministic model — proves determinism so the next assertion means
something), and only then (c) asserts the t1-side derived quantities (mask
stats, per-class delta, text) actually differ.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
import rasterio
import torch
from PIL import Image

from satquery.contracts import CONFIDENCE_BASES, validate_change_result
from satquery.io.modality import ModalityDecision
from satquery.io.raster import RasterInput
from satquery.specialists.change import (
    _MIN_GOOD_MATCHES,
    _already_coregistered,
    _orb_register,
    _per_class_delta,
    _softmax,
    run_change,
)

CUDA_AVAILABLE = torch.cuda.is_available()

_S2_BASE = "data/bigearthnet/images/BigEarthNet-S2"
_CHECKPOINT = "checkpoints/benclip/benclip_state.pt"

# Three real, distinct BigEarthNet S2 patches (B04, single band each — BEN
# stores one band per GeoTIFF, PLAN.md §4.5). A and B share a tile/CRS but
# have different pixel-grid transforms (adjacent patches); C is a wholly
# different acquisition (different date, different CRS/UTM zone).
_SCENE_A = os.path.join(
    _S2_BASE,
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP",
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_61_39",
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_61_39_B04.tif",
)
_SCENE_B = os.path.join(
    _S2_BASE,
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP",
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_61_42",
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_61_42_B04.tif",
)
_SCENE_C = os.path.join(
    _S2_BASE,
    "S2A_MSIL2A_20170617T113321_N9999_R080_T29UPU",
    "S2A_MSIL2A_20170617T113321_N9999_R080_T29UPU_00_38",
    "S2A_MSIL2A_20170617T113321_N9999_R080_T29UPU_00_38_B04.tif",
)

_data_available = all(os.path.exists(p) for p in (_SCENE_A, _SCENE_B, _SCENE_C))
_checkpoint_available = os.path.exists(_CHECKPOINT)

# Every test that calls the real run_change dispatches benclip internally.
# Gated behind CUDA + BEN data + checkpoint so a GPU-less/data-less clone
# still gets a green suite (PLAN.md §5.2 corollary 2) — mirrors
# tests/test_w5_fusion.py's `requires_data`.
requires_data = pytest.mark.skipif(
    not CUDA_AVAILABLE or not _data_available or not _checkpoint_available,
    reason="requires CUDA + BigEarthNet S2 data + a benclip checkpoint on disk",
)


def _write_geotiff(path: str, arr: np.ndarray, crs, transform) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype.name,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(arr, 1)


def _fake_raster(is_georeferenced, crs, transform, shape=(10, 10)) -> RasterInput:
    """A minimal RasterInput for pure-logic tests of `_already_coregistered`
    that need no real file and no model."""
    return RasterInput(
        array=np.zeros((*shape, 1), dtype=np.uint8),
        modality="optical",
        band_count=1,
        band_names=["Band_1"],
        crs=crs,
        transform=transform,
        is_georeferenced=is_georeferenced,
        source_path="fake.tif",
        display_rgb=np.zeros((*shape, 3), dtype=np.uint8),
        modality_decision=ModalityDecision(
            modality="optical", mechanism="band_count", reason="test fixture"
        ),
    )


# ---------------------------------------------------------------------------
# Pure-logic unit tests: no CUDA, no data, no model dispatch.
# ---------------------------------------------------------------------------


def test_already_coregistered_true_for_identical_geo():
    t = rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    r0 = _fake_raster(True, "EPSG:32633", t)
    r1 = _fake_raster(True, "EPSG:32633", t)
    assert _already_coregistered(r0, r1) is True


def test_already_coregistered_false_when_transform_differs():
    """Same CRS, different transform: must NOT be treated as co-registered —
    this is the exact case PLAN.md W4 task 1 requires reading the transform
    for, not just the CRS."""
    t0 = rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    t1 = rasterio.Affine(10.0, 0.0, 100.0, 0.0, -10.0, 0.0)
    r0 = _fake_raster(True, "EPSG:32633", t0)
    r1 = _fake_raster(True, "EPSG:32633", t1)
    assert _already_coregistered(r0, r1) is False


def test_already_coregistered_false_when_not_georeferenced():
    r0 = _fake_raster(False, None, None)
    r1 = _fake_raster(False, None, None)
    assert _already_coregistered(r0, r1) is False


def test_already_coregistered_false_different_crs():
    t = rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    r0 = _fake_raster(True, "EPSG:32633", t)
    r1 = _fake_raster(True, "EPSG:4326", t)
    assert _already_coregistered(r0, r1) is False


def test_softmax_sums_to_one_and_is_positive():
    p = _softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(float(p.sum()) - 1.0) < 1e-9
    assert (p > 0).all()


def test_softmax_handles_degenerate_all_equal_input():
    p = _softmax(np.array([0.0, 0.0, 0.0]))
    assert abs(float(p.sum()) - 1.0) < 1e-6
    assert np.allclose(p, 1.0 / 3.0)


def test_per_class_delta_sorted_by_magnitude_and_covers_union():
    labels_t0 = [{"label": "water", "score": 0.5}, {"label": "urban", "score": 0.1}]
    labels_t1 = [{"label": "water", "score": 0.1}, {"label": "urban", "score": 0.5}]
    deltas = _per_class_delta(labels_t0, labels_t1)
    assert {d["label"] for d in deltas} == {"water", "urban"}
    assert abs(deltas[0]["delta_confidence_pct"]) >= abs(deltas[1]["delta_confidence_pct"])
    # confidence percentages sum to ~100 on each date (softmax normalised)
    assert abs(sum(d["confidence_pct_t0"] for d in deltas) - 100.0) < 1e-6
    assert abs(sum(d["confidence_pct_t1"] for d in deltas) - 100.0) < 1e-6


def test_per_class_delta_handles_label_only_in_one_date():
    """A class predicted in one date but absent from the other's returned
    labels must not be dropped or crash — it gets a raw score of 0.0 for the
    missing date, per the module's documented policy."""
    labels_t0 = [{"label": "water", "score": 0.5}]
    labels_t1 = [{"label": "urban", "score": 0.5}]
    deltas = _per_class_delta(labels_t0, labels_t1)
    assert {d["label"] for d in deltas} == {"water", "urban"}


def test_per_class_delta_empty_on_no_labels():
    assert _per_class_delta([], []) == []


def test_orb_register_guards_empty_descriptors_no_crash():
    """PLAN.md §2.5 bug 3: old_files/BT_CM.py crashes here (bf.match on None
    descriptors). Uniform + tiny noise is measured (docs/status/W4.md) to
    yield ZERO ORB keypoints at this amplitude."""
    rng = np.random.default_rng(1)
    g0 = np.clip(rng.normal(130, 2, size=(64, 64)), 0, 255).astype(np.uint8)
    g1 = np.clip(rng.normal(132, 2, size=(64, 64)), 0, 255).astype(np.uint8)
    result = _orb_register(g0, g1)
    assert result["mechanism"] == "no_warp_fallback"
    assert result["confidence"] < 0.2
    assert result["aligned"].shape == g0.shape


def test_orb_register_recovers_a_known_affine_on_textured_image():
    """Positive control for the warp path: a textured synthetic image
    warped by a KNOWN small rotation+translation should be registered back,
    reducing the pixel difference substantially. Also exercises the §2.5
    bug-1 fix (fitting on the FILTERED `good` matches) on a case where a
    real affine actually exists to recover."""
    rng = np.random.default_rng(3)
    h, w = 200, 200
    g0 = np.zeros((h, w), dtype=np.uint8)
    for _ in range(40):
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        radius = int(rng.integers(5, 15))
        cv2.circle(g0, (x, y), radius, int(rng.integers(50, 255)), -1)
    g0 = cv2.GaussianBlur(g0, (3, 3), 0)

    known_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.0)
    known_matrix[:, 2] += [5, -4]
    g1 = cv2.warpAffine(g0, known_matrix, (w, h))

    result = _orb_register(g0, g1)
    assert result["mechanism"] == "orb_affine"
    assert result["good_matches"] >= _MIN_GOOD_MATCHES

    diff_before = float(np.mean(cv2.absdiff(g0, g1)))
    diff_after = float(np.mean(cv2.absdiff(g0, result["aligned"])))
    assert diff_after < diff_before * 0.5, (
        f"registration should substantially reduce pixel diff: "
        f"before={diff_before:.2f} after={diff_after:.2f}"
    )


def test_run_change_missing_files_does_not_crash():
    """Neither file exists -> must degrade to a valid, honest error result
    (never dispatches benclip, so this needs no CUDA/data gate)."""
    result = run_change("does/not/exist_t0.tif", "does/not/exist_t1.tif", "what changed?")
    validate_change_result(result)
    assert result["confidence_basis"] in CONFIDENCE_BASES
    assert result["change_mask_path"] is None
    assert result["overlay_path"] is None
    assert "error" in result["evidence"]


def test_featureless_pair_reports_low_confidence_not_crash(tmp_path):
    """PLAN.md W4 task 7: uniform water/farmland is exactly where ORB dies.
    Saved as PNG (no georeferencing) so the skip-registration path cannot
    fire and the guarded ORB fallback is genuinely exercised end to end
    through run_change, not just `_orb_register` in isolation."""
    rng = np.random.default_rng(7)
    h, w = 120, 120
    t0 = np.clip(rng.normal(130, 2, size=(h, w)), 0, 255).astype(np.uint8)
    t1 = np.clip(rng.normal(132, 2, size=(h, w)), 0, 255).astype(np.uint8)

    p0 = str(tmp_path / "featureless_t0.png")
    p1 = str(tmp_path / "featureless_t1.png")
    Image.fromarray(t0, mode="L").save(p0)
    Image.fromarray(t1, mode="L").save(p1)

    result = run_change(p0, p1, "what changed?")  # must not raise
    validate_change_result(result)
    assert result["evidence"]["registration"]["mechanism"] == "no_warp_fallback"
    assert result["metrics"]["registration_confidence"] < 0.2


# ---------------------------------------------------------------------------
# Real-imagery mechanics + synthetic-but-honest correctness (gated).
# ---------------------------------------------------------------------------


@requires_data
def test_run_change_real_scenes_contract_valid_with_mask_and_overlay():
    """Mechanics test on real imagery — two DIFFERENT real BigEarthNet S2
    patches (never claimed as a genuine bi-temporal pair, see module
    docstring). Exercises the full path: registration decision,
    differencing, benclip on both dates, mask+overlay written to disk."""
    result = run_change(_SCENE_A, _SCENE_B, "what changed between these two dates?")
    validate_change_result(result)
    assert result["confidence_basis"] == "heuristic"
    assert result["change_mask_path"] is not None
    assert os.path.exists(result["change_mask_path"])
    assert result["overlay_path"] is not None
    assert os.path.exists(result["overlay_path"])
    assert "registration_confidence" in result["metrics"]
    assert "changed_area_fraction" in result["metrics"]
    assert result["evidence"]["per_class_delta"], "benclip must run on both real dates"
    assert result["evidence"]["t0_modality"] and result["evidence"]["t1_modality"]


@requires_data
def test_registration_skips_when_identical_geo():
    """Same file both dates -> identical CRS/transform -> the skip path is
    taken, not ORB (PLAN.md W4 task 1)."""
    result = run_change(_SCENE_A, _SCENE_A, "what changed?")
    assert result["evidence"]["registration"]["mechanism"] == "skip_already_coregistered"
    assert result["metrics"]["registration_confidence"] == 1.0
    assert result["metrics"]["changed_area_fraction"] == 0.0


@requires_data
def test_registration_does_not_skip_when_transform_differs():
    """Same tile/CRS, adjacent patch -> different transform -> must NOT take
    the skip path (proves the decision reads the transform, not just CRS)."""
    result = run_change(_SCENE_A, _SCENE_B, "what changed?")
    mechanism = result["evidence"]["registration"]["mechanism"]
    assert mechanism != "skip_already_coregistered"
    assert mechanism in ("orb_affine", "no_warp_fallback")


@requires_data
def test_synthetic_known_change_recovered_by_mask(tmp_path):
    """SYNTHETIC-BUT-HONEST correctness fixture (PLAN.md W4 brief): t0 is a
    REAL BigEarthNet S2 B04 patch's raw pixels; t1 is that SAME array with a
    KNOWN rectangular region raised by a KNOWN, documented amount. This is
    not a real bi-temporal pair — it is a real scene plus a scripted,
    known edit — so the mask's recovery of exactly that region is a real
    correctness signal, not a comparison to ground truth we don't have."""
    with rasterio.open(_SCENE_A) as src:
        base = src.read(1)
        crs, transform = src.crs, src.transform

    y0, y1, x0, x1 = 40, 80, 40, 80  # the KNOWN changed region
    lo, hi = int(base.min()), int(base.max())
    delta = int((hi - lo) * 0.3)
    t1 = base.copy()
    t1[y0:y1, x0:x1] = np.clip(
        t1[y0:y1, x0:x1].astype(np.int64) + delta, lo, hi
    ).astype(base.dtype)

    p0 = str(tmp_path / "known_t0.tif")
    p1 = str(tmp_path / "known_t1.tif")
    _write_geotiff(p0, base, crs, transform)
    _write_geotiff(p1, t1, crs, transform)

    result = run_change(p0, p1, "what changed?")
    assert result["evidence"]["registration"]["mechanism"] == "skip_already_coregistered"

    mask = np.array(Image.open(result["change_mask_path"]))
    gt = np.zeros(base.shape, dtype=bool)
    gt[y0:y1, x0:x1] = True
    pred = mask > 0

    recall = float(pred[gt].sum()) / float(gt.sum())
    false_positive_fraction = float(pred[~gt].sum()) / float((~gt).sum())

    # Measured on this exact fixture at delta=30% of dynamic range: recall
    # ~0.99, false-positive fraction ~0.015 (docs/status/W4.md). Thresholds
    # below leave margin without pretending precision we haven't measured.
    assert recall > 0.85, f"mask recovered only {recall:.2%} of the known-changed region"
    assert false_positive_fraction < 0.05, (
        f"mask flagged {false_positive_fraction:.2%} of the UNCHANGED region as changed"
    )
    assert result["metrics"]["changed_area_fraction"] > 0.0


@requires_data
def test_swap_t1_changes_result_on_derived_quantities_only():
    """THE TEST THAT MATTERS MOST (PLAN.md W4 brief). t0 held fixed at
    _SCENE_A; t1 swapped between two different real scenes (_SCENE_B,
    _SCENE_C). Proves both dates are genuinely read, resting ONLY on
    derived quantities — never on an echoed input path (see module
    docstring for why that would be vacuous, per W5's original mistake)."""
    result_b = run_change(_SCENE_A, _SCENE_B, "what changed?")
    result_c = run_change(_SCENE_A, _SCENE_C, "what changed?")

    # Precondition: benclip actually produced a delta on BOTH calls, or the
    # comparisons below would be comparing [] to [] and prove nothing.
    assert result_b["evidence"]["per_class_delta"], "no t1=B delta — swap test would be vacuous"
    assert result_c["evidence"]["per_class_delta"], "no t1=C delta — swap test would be vacuous"

    # t0 side must be IDENTICAL across both calls (same file, deterministic
    # model) — this is what makes the difference below attributable to t1
    # specifically, not to nondeterminism.
    assert result_b["evidence"]["benclip_t0_labels"] == result_c["evidence"]["benclip_t0_labels"]

    # t1 side / derived outputs must differ.
    assert result_b["evidence"]["per_class_delta"] != result_c["evidence"]["per_class_delta"]
    assert result_b["evidence"]["mask_stats"] != result_c["evidence"]["mask_stats"]
    assert result_b["metrics"]["changed_area_fraction"] != result_c["metrics"]["changed_area_fraction"]
    assert result_b["text_response"] != result_c["text_response"]


@requires_data
def test_change_via_controller_yields_two_models_used(tmp_path):
    """PLAN.md W4 DONE MEANS #7: a change run through the real orchestrator
    shows >=2 models_used entries in the trace (benclip + the VLM role that
    verbalises, per satquery.controller.trace.TASK_ROLES["change"]). Uses
    the real controller read-only — satquery/controller/** is W6-owned."""
    from satquery.controller.trace import run_query

    trace = run_query(
        "What changed between these two dates, and where did the change occur?",
        [_SCENE_A, _SCENE_B],
        runs_root=str(tmp_path / "runs"),
    )
    assert trace["task_selected"] == "change"
    assert len(trace["models_used"]) >= 2
    assert trace["result"]["confidence_basis"] in CONFIDENCE_BASES
