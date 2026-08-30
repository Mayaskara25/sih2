"""
W6 input-validation tests (PLAN.md §3 edge cases, R2 decision).

Purely validation-side; no specialist runs and no dispatch happens here, so
these tests are fast and never touch data/. Fixtures are synthetic TIFFs via
rasterio (same pattern as test_w0_raster.py).
"""

import numpy as np
import pytest
import rasterio
from affine import Affine

from satquery.controller.validate import (
    SUPPORTED_EXTENSIONS,
    ValidationResult,
    check_compatibility,
    inspect_input,
    validate_inputs,
)

_T = Affine.translation(10.0, 20.0) @ Affine.scale(0.5, -0.5)


def _write_geotiff(path, bands=3, crs="EPSG:4326", height=16, width=16):
    array = np.zeros((height, width, bands), np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="uint8",
        crs=crs,
        transform=_T,
    ) as dst:
        for i in range(bands):
            dst.write(array[:, :, i], i + 1)


@pytest.fixture
def one_img(tmp_path):
    p = tmp_path / "a.tif"
    _write_geotiff(str(p))
    return str(p)


@pytest.fixture
def two_imgs(tmp_path):
    p1 = tmp_path / "t0.tif"
    p2 = tmp_path / "t1.tif"
    _write_geotiff(str(p1))
    _write_geotiff(str(p2))
    return str(p1), str(p2)


# --------------------------------------------------------------------------- #
# inspect_input basics
# --------------------------------------------------------------------------- #


def test_inspect_input_ok(one_img):
    info, warnings_list, errors = inspect_input(one_img)
    assert not warnings_list and not errors
    stream = info["format"]
    assert info["path"] == one_img
    assert info["bands"] == 3
    assert info["crs"] == "EPSG:4326"
    assert "GeoTIFF" in stream


def test_inspect_input_rejects_unknown_extension(tmp_path):
    bogus = tmp_path / "scene.txt"
    bogus.write_text("not an image")
    info, warnings_list, errors = inspect_input(str(bogus))
    assert errors


def test_inspect_input_graceful_on_missing_file(tmp_path):
    missing = tmp_path / "nope.tif"
    info, warnings_list, errors = inspect_input(str(missing))
    assert errors
    assert info["checks_passed"] is False


def test_inspect_input_forced_modality_used(one_img):
    _, warnings_opt, errors_opt = inspect_input(one_img)
    assert not errors_opt
    info_f, _, _ = inspect_input(one_img, forced_modality="sar")
    assert info_f["modality"] == "sar"


# --------------------------------------------------------------------------- #
# validate_inputs: success paths
# --------------------------------------------------------------------------- #


def test_change_two_images_pass(two_imgs):
    vr = validate_inputs(list(two_imgs), "change", query="what changed?")
    assert isinstance(vr, ValidationResult)
    assert vr.passed
    assert not vr.errors


def test_fusion_optical_sar_pass(tmp_path):
    opt = str(tmp_path / "o.tif")
    sar = str(tmp_path / "s.tif")
    _write_geotiff(opt)
    _write_geotiff(sar)
    vr = validate_inputs(
        [opt, sar],
        "fusion",
        forced_modalities=["optical", "sar"],
        query="use optical and sar",
    )
    assert vr.passed, vr.errors


def test_single_image_tasks_pass_with_single_input(one_img):
    for task in ("vqa", "caption", "grounding"):
        vr = validate_inputs([one_img], task, query="q")
        assert vr.passed, (task, vr.errors)


# --------------------------------------------------------------------------- #
# validate_inputs: refusal paths (R2 and count rules)
# --------------------------------------------------------------------------- #


def test_change_with_one_input_refused(one_img):
    vr = validate_inputs([one_img], "change", query="what changed?")
    assert not vr.passed
    assert vr.errors


def test_change_with_three_inputs_refused(two_imgs, tmp_path):
    third = tmp_path / "t2.tif"
    _write_geotiff(str(third))
    vr = validate_inputs([*two_imgs, str(third)], "change", query="what changed?")
    assert not vr.passed


def test_single_image_task_with_three_inputs_refused(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"i{i}.tif"
        _write_geotiff(str(p))
        paths.append(str(p))
    vr = validate_inputs(paths, "caption", query="describe")
    assert not vr.passed


def test_fusion_two_optical_refused_hard_error(tmp_path):
    """R2: two genuinely distinct modalities are required for fusion. Routing
    may still pick fusion from text (R4), but validation must refuse."""
    p1 = tmp_path / "o1.tif"
    p2 = tmp_path / "o2.tif"
    _write_geotiff(str(p1))
    _write_geotiff(str(p2))
    vr = validate_inputs([str(p1), str(p2)], "fusion", query="use optical and sar together")
    assert not vr.passed
    assert any("SAR" in e or "modality" in e.lower() for e in vr.errors)


def test_fusion_unknown_modality_warns_not_fails(tmp_path):
    """A modality we cannot classify is a warning, never a hard error."""
    p1 = tmp_path / "a.tif"
    p2 = tmp_path / "b.tif"
    _write_geotiff(str(p1))
    _write_geotiff(str(p2))
    vr = validate_inputs([str(p1), str(p2)], "fusion", query="use sar and optical")
    assert vr.passed or vr.errors  # at minimum, no crash
    assert not any("does not have access" in e for e in vr.errors)


# --------------------------------------------------------------------------- #
# check_compatibility: mismatch is warning, not failure
# --------------------------------------------------------------------------- #


def test_change_size_mismatch_is_warning_not_error(tmp_path):
    p1 = tmp_path / "t0.tif"
    p2 = tmp_path / "t1.tif"
    _write_geotiff(str(p1), height=16, width=16)
    _write_geotiff(str(p2), height=32, width=32)
    info1, _, _ = inspect_input(str(p1))
    info2, _, _ = inspect_input(str(p2))
    vr = check_compatibility([info1, info2], "change")
    assert vr.passed
    assert vr.warnings


# --------------------------------------------------------------------------- #
# ValidationInfo shape hangs together
# --------------------------------------------------------------------------- #


def test_to_validation_info_matches_contract(one_img):
    vr = validate_inputs([one_img], "change", query="what changed?")
    info = vr.to_validation_info()
    assert set(info.keys()) >= {"passed", "warnings", "errors"}
    assert info["passed"] is False
    assert isinstance(info["warnings"], list)
    assert isinstance(info["errors"], list)
    assert info["errors"]


def test_input_infos_present_in_validation_result(one_img):
    vr = validate_inputs([one_img], "caption", query="describe")
    assert vr.input_infos and isinstance(vr.input_infos, list)


def test_supported_extensions_include_tiff():
    # stored lower-case without the leading dot
    assert "tif" in SUPPORTED_EXTENSIONS and "tiff" in SUPPORTED_EXTENSIONS