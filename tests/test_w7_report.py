"""W7 tests: report generation and Streamlit-app helpers (PLAN.md W7).

Tests are kept on logic functions that live outside Streamlit's API surface.
The Streamlit main() is inherently integration-level; we verify the helpers
it calls are correct, and that the report generator produces valid HTML.

The report and the app helpers exercise the *real* controller (run_query)
through stubs — the same stubs W0 shipped — so no model downloads, no GPU,
and no data/ files are touched.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from satquery.contracts import validate_execution_trace
from satquery.controller.trace import run_query
from satquery.report import format_confidence_basis, generate_report

# ---------------------------------------------------------------------------
# Fixtures — synthetic GeoTIFFs (same pattern as W6 tests)
# ---------------------------------------------------------------------------

_T = Affine.translation(10.0, 20.0) @ Affine.scale(0.5, -0.5)


def _write_geotiff(path, bands=3, crs="EPSG:4326", height=16, width=16):
    array = np.random.default_rng(42).integers(0, 256, (height, width, bands), np.uint8)
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
    p = tmp_path / "optical.tif"
    _write_geotiff(str(p))
    return str(p)


@pytest.fixture
def two_imgs(tmp_path):
    p1 = tmp_path / "optical.tif"
    p2 = tmp_path / "sar.tif"
    _write_geotiff(str(p1), bands=3)
    _write_geotiff(str(p2), bands=1)
    return str(p1), str(p2)


@pytest.fixture
def vqa_trace(tmp_path, one_img):
    return run_query(
        "How many buildings are visible?",
        [one_img],
        runs_root=str(tmp_path / "runs"),
    )


@pytest.fixture
def change_trace(tmp_path, two_imgs):
    return run_query(
        "What changed between these two dates?",
        list(two_imgs),
        runs_root=str(tmp_path / "runs"),
    )


@pytest.fixture
def fusion_trace(tmp_path, two_imgs):
    return run_query(
        "Use the optical and SAR images together.",
        list(two_imgs),
        forced_modalities=["optical", "sar"],
        runs_root=str(tmp_path / "runs"),
    )


@pytest.fixture
def validation_fail_trace(tmp_path, one_img):
    return run_query(
        "What changed between these two dates?",
        [one_img],
        runs_root=str(tmp_path / "runs"),
    )


# ---------------------------------------------------------------------------
# Tests: report.format_confidence_basis
# ---------------------------------------------------------------------------


class TestFormatConfidenceBasis:
    def test_stub(self):
        assert "placeholder" in format_confidence_basis("stub", 0.0).lower()

    def test_heuristic(self):
        result = format_confidence_basis("heuristic", 0.75)
        assert "0.75" in result
        assert "heuristic" in result.lower()

    def test_calibrated(self):
        result = format_confidence_basis("calibrated", 0.92)
        assert "0.92" in result
        assert "calibrated" in result.lower()

    def test_model_logprob(self):
        result = format_confidence_basis("model_logprob", 0.85)
        assert "0.85" in result
        assert "log-probability" in result.lower()

    def test_unknown_basis(self):
        result = format_confidence_basis("unknown_basis", 0.5)
        assert "0.50" in result
        assert "unknown_basis" in result


# ---------------------------------------------------------------------------
# Tests: report.generate_report — produces valid HTML with expected sections
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_vqa_report_contains_all_sections(self, vqa_trace, tmp_path):
        report_path = str(tmp_path / "report.html")
        result = generate_report(vqa_trace, report_path)
        assert result == report_path

        html_text = Path(report_path).read_text(encoding="utf-8")
        assert vqa_trace["query"] in html_text
        assert vqa_trace["run_id"] in html_text
        assert vqa_trace["task_selected"] in html_text
        assert "Execution Report" in html_text
        assert "Full Execution Trace" in html_text
        assert "Models Used" in html_text or "Models" in html_text

    def test_change_report_contains_trace(self, change_trace, tmp_path):
        report_path = str(tmp_path / "report.html")
        generate_report(change_trace, report_path)
        html_text = Path(report_path).read_text(encoding="utf-8")
        # The report HTML-escapes the trace JSON; verify key fields survive
        assert change_trace["run_id"] in html_text
        assert change_trace["query"] in html_text
        assert change_trace["task_selected"] in html_text
        assert "Full Execution Trace" in html_text

    def test_fusion_report_contains_trace(self, fusion_trace, tmp_path):
        report_path = str(tmp_path / "report.html")
        generate_report(fusion_trace, report_path)
        html_text = Path(report_path).read_text(encoding="utf-8")
        assert "fusion" in html_text.lower()

    def test_validation_failure_report(self, validation_fail_trace, tmp_path):
        report_path = str(tmp_path / "report.html")
        generate_report(validation_fail_trace, report_path)
        html_text = Path(report_path).read_text(encoding="utf-8")
        assert "validation_failed" in html_text.lower() or "error" in html_text.lower()

    def test_report_is_self_contained(self, vqa_trace, tmp_path):
        """Report HTML must not reference external CSS/JS/images."""
        report_path = str(tmp_path / "report.html")
        generate_report(vqa_trace, report_path)
        html_text = Path(report_path).read_text(encoding="utf-8")
        assert "<link rel=\"stylesheet\"" not in html_text
        assert "<script src=" not in html_text
        # All images must be data URIs or absent
        assert "data:image/" in html_text or "<img" not in html_text


# ---------------------------------------------------------------------------
# Tests: controller round-trip through the full pipeline (end-to-end)
# ---------------------------------------------------------------------------


class TestControllerRoundTrips:
    def test_single_image_vqa_round_trip(self, vqa_trace):
        validate_execution_trace(vqa_trace)
        assert vqa_trace["task_selected"] == "vqa"
        assert vqa_trace["validation"]["passed"] is True
        assert vqa_trace["result"].get("status") != "validation_failed"
        assert len(vqa_trace["models_used"]) >= 1
        # Stub specialist was called
        assert vqa_trace["result"]["confidence_basis"] == "stub"

    def test_two_image_change_round_trip(self, change_trace):
        validate_execution_trace(change_trace)
        assert change_trace["task_selected"] == "change"
        assert change_trace["validation"]["passed"] is True
        assert len(change_trace["models_used"]) >= 2

    def test_fusion_round_trip(self, fusion_trace):
        validate_execution_trace(fusion_trace)
        assert fusion_trace["task_selected"] == "fusion"
        assert fusion_trace["validation"]["passed"] is True
        assert len(fusion_trace["models_used"]) >= 2

    def test_validation_refusal_round_trip(self, validation_fail_trace):
        validate_execution_trace(validation_fail_trace)
        assert validation_fail_trace["validation"]["passed"] is False
        assert validation_fail_trace["result"]["status"] == "validation_failed"


# ---------------------------------------------------------------------------
# Tests: app.main helper functions (pure logic, no Streamlit API)
# ---------------------------------------------------------------------------


class TestAppHelpers:
    def test_resolve_image_info(self, one_img):
        from app.main import _resolve_image_info

        info = _resolve_image_info(one_img, user_modality=None)
        assert info["band_count"] == 3
        assert info["modality"] in ("optical", "msi", "sar", "unknown")
        assert info["mechanism"] in ("user", "filename", "metadata", "band_count")
        assert isinstance(info["display_rgb"], np.ndarray)
        assert info["display_rgb"].shape[2] == 3
        assert info["display_rgb"].dtype == np.uint8
        assert info["raw_path"] == one_img

    def test_resolve_image_info_with_user_override(self, one_img):
        from app.main import _resolve_image_info

        info = _resolve_image_info(one_img, user_modality="sar")
        assert info["modality"] == "sar"
        assert info["mechanism"] == "user"

    def test_resolve_image_info_single_band(self, tmp_path):
        from app.main import _resolve_image_info

        p = tmp_path / "single_band.tif"
        _write_geotiff(str(p), bands=1)
        info = _resolve_image_info(str(p), user_modality=None)
        assert info["band_count"] == 1

    def test_render_confidence_stub(self):
        from app.main import render_confidence

        result = {"confidence": 0.0, "confidence_basis": "stub"}
        out = render_confidence(result)
        assert "0.00" in out
        assert "placeholder" in out.lower()

    def test_render_confidence_heuristic(self):
        from app.main import render_confidence

        result = {"confidence": 0.75, "confidence_basis": "heuristic"}
        out = render_confidence(result)
        assert "0.75" in out
        assert "heuristic" in out.lower()

    def test_render_confidence_no_confidence(self):
        from app.main import render_confidence

        result = {}
        out = render_confidence(result)
        assert out == "N/A"

    def test_get_example_queries(self):
        from app.main import get_example_queries

        examples = get_example_queries()
        assert len(examples) == 5
        labels = [label for label, _ in examples]
        assert "VQA" in labels
        assert "Change" in labels
        assert "Fusion" in labels

    def test_save_upload(self, tmp_path, monkeypatch):
        """Test the upload-to-temp-file helper with a mock UploadedFile."""
        from app.main import _save_upload

        class FakeUpload:
            def __init__(self, data, name="test.tif"):
                self._data = data
                self.name = name

            def read(self):
                return self._data

        fake = FakeUpload(b"fake tif data", name="test.tif")
        path = _save_upload(fake, suffix=".tif")
        assert os.path.isfile(path)
        assert Path(path).suffix == ".tif"
        with open(path, "rb") as f:
            assert f.read() == b"fake tif data"
        os.unlink(path)


# ---------------------------------------------------------------------------
# Tests: modality selector demonstrably changes behaviour (W7 acceptance #5)
# ---------------------------------------------------------------------------


class TestModalitySelectorBehaviour:
    def test_two_optical_to_fusion_is_refused(self, tmp_path):
        """Sending two optical images to fusion via user override produces
        the R2 validation refusal — the modality selector changes behaviour."""
        p1 = tmp_path / "opt_a.tif"
        p2 = tmp_path / "opt_b.tif"
        _write_geotiff(str(p1), bands=3)
        _write_geotiff(str(p2), bands=3)
        trace = run_query(
            "Use the optical and SAR images together.",
            [str(p1), str(p2)],
            forced_modalities=["optical", "optical"],
            runs_root=str(tmp_path / "runs"),
        )
        validate_execution_trace(trace)
        assert trace["validation"]["passed"] is False
        errors = trace["validation"]["errors"]
        assert any("optical" in e.lower() or "distinct" in e.lower() for e in errors)

    def test_auto_detect_vs_user_override(self, tmp_path):
        """User override (tier 1) takes precedence over auto-detection."""
        from app.main import _resolve_image_info

        p = tmp_path / "img.tif"
        _write_geotiff(str(p), bands=3)
        auto_info = _resolve_image_info(str(p), user_modality=None)
        override_info = _resolve_image_info(str(p), user_modality="sar")
        # Auto-detect picks up filename or band-count; user override always wins
        assert override_info["mechanism"] == "user"
        assert override_info["modality"] == "sar"
        # The auto info may differ (optical from band count)
        assert auto_info["mechanism"] != "user"
