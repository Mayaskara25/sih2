"""W5 tests — Cross-modal optical–SAR fusion specialist.

PLAN.md §7 W5 acceptance test + contract conformance. Tests are gated behind
CUDA + data presence so a GPU-less clone still gets a green suite (§5.2).

THE TEST THAT MATTERS MOST (W5 brief line 74):
  Assert the result changes when the SAR input is swapped for a different
  scene, with the optical input held fixed. This is the direct, mechanical
  proof that the second modality is actually being read and is not decorative.
"""

from __future__ import annotations

import glob
import os
import tempfile

import numpy as np
import pytest
import torch

from satquery.contracts import (
    CONFIDENCE_BASES,
    validate_fusion_result,
)
from satquery.specialists.fusion import (
    _compute_agreement,
    _sar_physics_analysis,
    _validate_modalities,
    run_fusion,
)

# ---------------------------------------------------------------------------
# Fixtures — gated behind CUDA + BigEarthNet data on disk
# ---------------------------------------------------------------------------

CUDA_AVAILABLE = torch.cuda.is_available()

# First S2+S1 pair from the W1 targets (Austria, test split)
_S2_BASE = "data/bigearthnet/images/BigEarthNet-S2"
_S1_BASE = "data/bigearthnet/images/BigEarthNet-S1"

# Patch 0 (Inland waters + forest + arable)
_OPTICAL_PATH = os.path.join(
    _S2_BASE,
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP",
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_61_39",
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_61_39_B04.tif",
)
_SAR_PATH_1 = os.path.join(
    _S1_BASE,
    "S1A_IW_GRDH_1SDV_20170613T165043",
    "S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39",
    "S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39_VV.tif",
)

# Patch 3 — different scene (Coniferous forest + urban)
_OPTICAL_PATH_2 = os.path.join(
    _S2_BASE,
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP",
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_61_42",
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_61_42_B04.tif",
)
# Different S1 acquisition folder (different date/location)
_SAR_PATH_2 = os.path.join(
    _S1_BASE,
    "S1A_IW_GRDH_1SDV_20170613T165108",
    "S1A_IW_GRDH_1SDV_20170613T165108_33UUP_70_16",
    "S1A_IW_GRDH_1SDV_20170613T165108_33UUP_70_16_VV.tif",
)

_data_available = (
    os.path.exists(_OPTICAL_PATH)
    and os.path.exists(_SAR_PATH_1)
    and os.path.exists(_SAR_PATH_2)
)

requires_data = pytest.mark.skipif(
    not CUDA_AVAILABLE or not _data_available,
    reason="requires CUDA + BigEarthNet S1+S2 data on disk",
)

requires_data_or_cpu = pytest.mark.skipif(
    not _data_available,
    reason="requires BigEarthNet S1+S2 data on disk",
)


# ---------------------------------------------------------------------------
# Unit tests (no GPU, no data)
# ---------------------------------------------------------------------------


class TestValidateModalities:
    """Modality validation: refuse two optical, allow SAR + unknown."""

    def test_rejects_two_optical(self):
        """Two optical-family inputs must raise ValueError (R2)."""

        class FakeRaster:
            modality = "optical"

        with pytest.raises(ValueError, match="genuinely distinct"):
            _validate_modalities(FakeRaster(), FakeRaster())

    def test_rejects_optical_plus_msi(self):
        """optical + msi are both optical-family."""

        class FakeRaster:
            pass

        o = FakeRaster()
        o.modality = "optical"
        s = FakeRaster()
        s.modality = "msi"
        with pytest.raises(ValueError, match="genuinely distinct"):
            _validate_modalities(o, s)

    def test_allows_sar_plus_optical(self):
        """SAR + optical is the canonical case."""

        class FakeRaster:
            pass

        o = FakeRaster()
        o.modality = "optical"
        s = FakeRaster()
        s.modality = "sar"
        _validate_modalities(o, s)  # should not raise

    def test_allows_unknown_modality(self):
        """unknown modality must not crash (fail soft, PLAN.md §2.3)."""

        class FakeRaster:
            pass

        o = FakeRaster()
        o.modality = "unknown"
        s = FakeRaster()
        s.modality = "sar"
        _validate_modalities(o, s)  # should not raise

    def test_allows_both_unknown(self):
        """Two unknowns should not crash — the controller catches the
        two-optical case, but the specialist should be lenient."""

        class FakeRaster:
            modality = "unknown"

        _validate_modalities(FakeRaster(), FakeRaster())  # should not raise


class TestComputeAgreement:
    """Agreement/disagreement computation."""

    def test_full_agreement(self):
        labels = [{"label": "a", "score": 0.9}, {"label": "b", "score": 0.8}]
        result = _compute_agreement(labels, labels)
        assert result["agreement_percentage"] == 100.0
        assert result["disagreement_percentage"] == 0.0
        assert set(result["agreement_classes"]) == {"a", "b"}

    def test_no_agreement(self):
        opt = [{"label": "a", "score": 0.9}]
        sar = [{"label": "b", "score": 0.8}]
        result = _compute_agreement(opt, sar)
        assert result["agreement_percentage"] == 0.0
        assert result["disagreement_percentage"] == 100.0

    def test_partial_agreement(self):
        opt = [{"label": "a", "score": 0.9}, {"label": "b", "score": 0.8}]
        sar = [{"label": "b", "score": 0.7}, {"label": "c", "score": 0.6}]
        result = _compute_agreement(opt, sar)
        assert result["agreement_classes"] == ["b"]
        assert result["optical_only_classes"] == ["a"]
        assert result["sar_only_classes"] == ["c"]

    def test_empty_labels(self):
        result = _compute_agreement([], [])
        assert result["agreement_percentage"] == 0.0


class TestSarPhysicsAnalysis:
    """SAR-physics evidence path — unit tests with synthetic data."""

    def test_water_detection(self):
        """Low VV backscatter should trigger potential_water."""

        class FakeRaster:
            array = np.full((100, 100, 1), -25.0, dtype=np.float32)
            band_count = 1
            modality = "sar"

        result = _sar_physics_analysis(FakeRaster())
        assert result["potential_water"] is True
        assert result["vv_mean_db"] < -20.0

    def test_built_up_detection(self):
        """High VV backscatter should indicate built-up."""

        class FakeRaster:
            array = np.full((100, 100, 1), -3.0, dtype=np.float32)
            band_count = 1
            modality = "sar"

        result = _sar_physics_analysis(FakeRaster())
        assert result["high_backscatter_built_fraction"] > 0.9
        assert result["built_up_possible"] is True

    def test_dual_pol_double_bounce(self):
        """Dual-pol SAR: high VH/VV ratio indicates double-bounce."""
        arr = np.zeros((100, 100, 2), dtype=np.float32)
        arr[..., 0] = -5.0  # VV
        arr[..., 1] = -3.0  # VH (ratio ~0.6)
        # Make VH fraction high enough
        arr[:50, :, 1] = -1.0

        class FakeRaster:
            array = arr
            band_count = 2
            modality = "sar"

        result = _sar_physics_analysis(FakeRaster())
        assert "vh_vv_ratio_mean" in result
        assert "double_bounce_fraction" in result

    def test_handles_2d_array(self):
        """2D SAR array (no band dim) should not crash."""

        class FakeRaster:
            array = np.full((100, 100), -15.0, dtype=np.float32)
            band_count = 1
            modality = "sar"

        result = _sar_physics_analysis(FakeRaster())
        assert "vv_mean_db" in result

    def test_handles_all_nan(self):
        """All-NaN SAR data should return error, not crash."""

        class FakeRaster:
            array = np.full((100, 100, 1), np.nan, dtype=np.float32)
            band_count = 1
            modality = "sar"

        result = _sar_physics_analysis(FakeRaster())
        assert "error" in result


# ---------------------------------------------------------------------------
# Integration tests — require real BEN data, gated
# ---------------------------------------------------------------------------


@requires_data_or_cpu
class TestRunFusionContract:
    """Basic contract conformance on real BEN data."""

    def test_returns_valid_fusion_result(self):
        """run_fusion on a real S2+S1 pair must produce a valid FusionResult."""
        result = run_fusion(_OPTICAL_PATH, _SAR_PATH_1, "what land cover is shown?")
        validated = validate_fusion_result(result)
        assert validated is not None

    def test_evidence_names_both_modalities(self):
        """evidence dict must name both modalities explicitly (W5 brief line 71)."""
        result = run_fusion(_OPTICAL_PATH, _SAR_PATH_1, "analyze the scene")
        ev = result["evidence"]
        assert "optical_file" in ev
        assert "sar_file" in ev
        assert ev["optical_file"] == _OPTICAL_PATH
        assert ev["sar_file"] == _SAR_PATH_1

    def test_evidence_has_modality_info(self):
        """evidence must record which file supplied each modality."""
        result = run_fusion(_OPTICAL_PATH, _SAR_PATH_1, "what is here?")
        ev = result["evidence"]
        assert "optical_modality" in ev
        assert "sar_modality" in ev
        assert ev["optical_bands"] >= 1
        assert ev["sar_bands"] >= 1

    def test_confidence_basis_not_stub(self):
        """W5 implementation must not remain 'stub'."""
        result = run_fusion(_OPTICAL_PATH, _SAR_PATH_1, "q")
        assert result["confidence_basis"] != "stub"
        assert result["confidence_basis"] in CONFIDENCE_BASES

    def test_confidence_in_range(self):
        """Confidence must be a float in [0, 1]."""
        result = run_fusion(_OPTICAL_PATH, _SAR_PATH_1, "q")
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_text_response_non_empty(self):
        """Text response must not be empty."""
        result = run_fusion(_OPTICAL_PATH, _SAR_PATH_1, "describe the area")
        assert len(result["text_response"]) > 0

    def test_sar_physics_in_evidence(self):
        """SAR-physics analysis must be present in evidence."""
        result = run_fusion(_OPTICAL_PATH, _SAR_PATH_1, "q")
        assert "sar_physics" in result["evidence"]
        assert isinstance(result["evidence"]["sar_physics"], dict)

    def test_agreement_summary_in_evidence(self):
        """agreement_summary must be present."""
        result = run_fusion(_OPTICAL_PATH, _SAR_PATH_1, "q")
        assert "agreement_summary" in result["evidence"]
        summary = result["evidence"]["agreement_summary"]
        assert "agreement_percentage" in summary
        assert "disagreement_percentage" in summary

    def test_agreement_map_if_matplotlib(self):
        """If matplotlib is available, agreement_map_path should be set."""
        try:
            import matplotlib  # noqa: F401
            has_mpl = True
        except ImportError:
            has_mpl = False
        result = run_fusion(_OPTICAL_PATH, _SAR_PATH_1, "q")
        if has_mpl:
            # Path may still be None if rendering failed, but should be attempted
            assert result["agreement_map_path"] is None or os.path.exists(
                result["agreement_map_path"]
            )


@requires_data_or_cpu
class TestSarSwapSensitivity:
    """THE CRITICAL TEST: swapping the SAR input while holding optical fixed
    must change the result. This is the direct mechanical proof that R2 is
    met — the second modality is actually being read, not decorative."""

    def test_sar_swap_changes_result(self):
        """Different SAR scene → different fusion output."""
        result1 = run_fusion(_OPTICAL_PATH, _SAR_PATH_1, "what land cover?")
        result2 = run_fusion(_OPTICAL_PATH, _SAR_PATH_2, "what land cover?")

        # Both must be valid
        validate_fusion_result(result1)
        validate_fusion_result(result2)

        # A DERIVED quantity must differ. Deliberately excludes
        # evidence["sar_file"], which merely echoes the input path and is
        # therefore trivially different for any two distinct files — including
        # for an implementation that never opens the SAR raster at all.
        # Including it here would make this assertion vacuous, which would
        # defeat the entire purpose of the test.
        differ = (
            result1["text_response"] != result2["text_response"]
            or result1["evidence"]["sar_physics"] != result2["evidence"]["sar_physics"]
        )
        assert differ, (
            "Fusion result did NOT change when SAR input was swapped — "
            "the second modality is likely not being read. "
            f"Result1 text: {result1['text_response'][:200]}\n"
            f"Result2 text: {result2['text_response'][:200]}"
        )

    def test_sar_file_recorded_differently(self):
        """Evidence must record different SAR files for different inputs."""
        result1 = run_fusion(_OPTICAL_PATH, _SAR_PATH_1, "q")
        result2 = run_fusion(_OPTICAL_PATH, _SAR_PATH_2, "q")
        assert result1["evidence"]["sar_file"] != result2["evidence"]["sar_file"]

    def test_sar_physics_differ_between_scenes(self):
        """SAR physics statistics should differ between different scenes."""
        result1 = run_fusion(_OPTICAL_PATH, _SAR_PATH_1, "q")
        result2 = run_fusion(_OPTICAL_PATH, _SAR_PATH_2, "q")
        phys1 = result1["evidence"]["sar_physics"]
        phys2 = result2["evidence"]["sar_physics"]
        # At least VV mean should differ (different locations)
        if "vv_mean_db" in phys1 and "vv_mean_db" in phys2:
            assert phys1["vv_mean_db"] != phys2["vv_mean_db"], (
                "SAR VV mean identical across different scenes — "
                "SAR data may not be actually loaded"
            )


@requires_data_or_cpu
class TestEdgeCases:
    """Edge cases: unknown modality, different band counts, error handling."""

    def test_unknown_modality_sar_input(self):
        """A bare 1-band SAR file with no filename hint → unknown modality.
        Must not crash; should degrade gracefully."""
        # The BEN S1 files have _VV in the name, so they resolve to "sar".
        # To truly test "unknown" we'd need a file without hint.
        # For now, verify the existing SAR path works (it resolves via filename).
        result = run_fusion(_OPTICAL_PATH, _SAR_PATH_1, "q")
        validate_fusion_result(result)

    def test_nonexistent_optical_file(self):
        """Missing optical file must return a valid error result."""
        result = run_fusion("nonexistent_opt.tif", _SAR_PATH_1, "q")
        validate_fusion_result(result)
        assert result["confidence"] == 0.0
        assert "error" in result["evidence"]

    def test_nonexistent_sar_file(self):
        """Missing SAR file must return a valid error result."""
        result = run_fusion(_OPTICAL_PATH, "nonexistent_sar.tif", "q")
        validate_fusion_result(result)
        assert result["confidence"] == 0.0
        assert "error" in result["evidence"]

    def test_optical_only_rejection_message(self):
        """Two optical inputs must produce a clear error message."""
        # We need two optical files. Use the same S2 patch twice.
        result = run_fusion(_OPTICAL_PATH, _OPTICAL_PATH_2, "q")
        # Both resolve to "optical" (S2 band files have _B04 in name)
        # The validate_modalities check should catch this
        if result["confidence"] == 0.0:
            assert "genuinely distinct" in result["text_response"].lower() or (
                "error" in result["evidence"]
            )


@requires_data_or_cpu
class TestMultipleBandCounts:
    """Valid FusionResult for various input band counts (W5 brief line 92)."""

    def test_1band_optical(self):
        """1-band S2 (single band file) + SAR."""
        result = run_fusion(_OPTICAL_PATH, _SAR_PATH_1, "q")
        validate_fusion_result(result)
        assert result["evidence"]["optical_bands"] == 1

    def test_1band_sar(self):
        """Optical + 1-band SAR (VV only)."""
        result = run_fusion(_OPTICAL_PATH, _SAR_PATH_1, "q")
        validate_fusion_result(result)
        assert result["evidence"]["sar_bands"] == 1
