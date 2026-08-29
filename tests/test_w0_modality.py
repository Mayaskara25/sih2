"""Unit tests for satquery.io.modality's precedence rules (PLAN.md §4.4).

These are pure-logic tests: resolve_modality takes plain Python values (a
path string, a band count int, a metadata dict) so none of this needs
rasterio or any real file on disk.
"""

import pytest

from satquery.io.modality import VALID_MODALITIES, ModalityDecision, resolve_modality


def test_valid_modalities_match_plan_contract():
    assert VALID_MODALITIES == ("optical", "msi", "sar", "unknown")


# ---------------------------------------------------------------------
# Tier 1: explicit user override always wins
# ---------------------------------------------------------------------

def test_user_override_beats_filename_and_band_count():
    decision = resolve_modality(
        user_modality="optical",
        source_path="RISAT_GRD_VV.tif",  # screams "sar" by filename
        band_count=1,  # and by band count could plausibly be sar too
    )
    assert decision.modality == "optical"
    assert decision.mechanism == "user"


def test_user_override_case_insensitive():
    decision = resolve_modality(user_modality="SAR", source_path="anything.tif", band_count=3)
    assert decision.modality == "sar"
    assert decision.mechanism == "user"


def test_user_override_invalid_raises():
    with pytest.raises(ValueError):
        resolve_modality(user_modality="hyperspectral", source_path="x.tif", band_count=3)


# ---------------------------------------------------------------------
# Tier 2: filename / metadata heuristics beat tier 3 band-count heuristic
# ---------------------------------------------------------------------

def test_filename_hint_beats_band_count_heuristic():
    # 3 bands alone would resolve to "optical" via band-count, but the
    # filename explicitly says SAR and must win.
    decision = resolve_modality(source_path="scene_RISAT_VV_VH.tif", band_count=3)
    assert decision.modality == "sar"
    assert decision.mechanism == "filename"


def test_filename_hint_optical():
    decision = resolve_modality(source_path="cartosat2s_panchromatic.tif", band_count=1)
    assert decision.modality == "optical"
    assert decision.mechanism == "filename"


def test_filename_hint_msi():
    decision = resolve_modality(source_path="sentinel2_tile_msi.tif", band_count=1)
    assert decision.modality == "msi"
    assert decision.mechanism == "filename"


def test_metadata_hint_used_when_no_filename_hint():
    decision = resolve_modality(
        source_path="scene_001.tif",
        band_count=1,
        metadata_tags={"SENSOR": "Sentinel-2 MSI"},
    )
    assert decision.modality == "msi"
    assert decision.mechanism == "metadata"


def test_filename_and_metadata_agree_reports_metadata_mechanism():
    decision = resolve_modality(
        source_path="risat_scene.tif",
        band_count=2,
        metadata_tags={"MODE": "SAR GRD"},
    )
    assert decision.modality == "sar"
    assert decision.mechanism == "metadata"


def test_filename_and_metadata_conflict_resolves_unknown():
    decision = resolve_modality(
        source_path="risat_scan.tif",  # filename says sar
        band_count=1,
        metadata_tags={"SENSOR": "Sentinel-2 MSI"},  # metadata says msi
    )
    assert decision.modality == "unknown"
    assert decision.mechanism == "metadata"
    assert "conflict" in decision.reason.lower()


# ---------------------------------------------------------------------
# Tier 3: band-count heuristic, only when nothing else is available
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "band_count,expected",
    [(3, "optical"), (4, "msi"), (12, "msi"), (13, "msi")],
)
def test_band_count_heuristic_expected_counts(band_count, expected):
    decision = resolve_modality(source_path="unlabeled_scene.tif", band_count=band_count)
    assert decision.modality == expected
    assert decision.mechanism == "band_count"


@pytest.mark.parametrize("band_count", [1, 2, 5, 7, 11, 14, 0])
def test_band_count_ambiguous_resolves_unknown(band_count):
    decision = resolve_modality(source_path="unlabeled_scene.tif", band_count=band_count)
    assert decision.modality == "unknown"
    assert decision.mechanism == "band_count"


def test_ambiguous_input_is_a_real_modality_decision_object():
    decision = resolve_modality(source_path="", band_count=1)
    assert isinstance(decision, ModalityDecision)
    assert decision.modality == "unknown"
    assert decision.reason  # never silent - must explain itself
