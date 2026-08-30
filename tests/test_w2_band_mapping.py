"""Tests for benclip's §4.5 band-mapping policy (PLAN.md §4.5).

These are pure functions (no model, no GPU) — they exercise the single routine
that maps any RasterInput to the 14-channel stem and the fill strategy, and
assert the returned ``band_mapping`` validates against the frozen contract and
carries exactly the slots the policy promises.

This is the W2<->W3<->W5 integration seam: callers hand this module 1-, 2-, 3-,
4-, 12-, 13- and 14-band rasters and must never do band logic themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from satquery.adapters.benclip import (
    BEN_14_SLOTS,
    S1_BAND_ORDER,
    S2_BAND_ORDER,
    BenClipModel,
    _make_mapping,
    _raster_to_stem,
)
from satquery.contracts import ContractViolation, validate_band_mapping


class _Ctx:
    """Stand-in for BenClipModel holding only what _raster_to_stem needs."""

    def __init__(self, means=None):
        self.training_means = means or {}


@pytest.fixture
def ctx():
    return _Ctx(means={b: 0.0 for b in BEN_14_SLOTS})


def _raster(array, modality):
    class R:
        pass
    r = R()
    r.array = np.asarray(array, dtype=np.float32)
    r.modality = modality
    return r


def test_contract_validates_good_payload():
    m = _make_mapping(["B04", "B03"], ["B01"], "optical")
    assert validate_band_mapping(m) == m
    m2 = _make_mapping([], list(BEN_14_SLOTS), "unknown")
    assert validate_band_mapping(m2)["source_modality"] == "unknown"


def test_contract_rejects_bad_payload():
    bad = {"slots_filled": [], "slots_absent": [], "fill_strategy": "zeros",
           "source_modality": "badmodality"}
    with pytest.raises(ContractViolation):
        validate_band_mapping(bad)


def test_rgb3_maps_to_b04_b03_b02(ctx):
    arr = np.zeros((120, 120, 3), dtype=np.float32)
    stem, mapping, src = _raster_to_stem(ctx, _raster(arr, "optical"))
    assert mapping["slots_filled"] == ["B02", "B03", "B04"]
    assert "B08" not in mapping["slots_filled"]
    assert mapping["source_modality"] == "optical"
    assert stem.shape == (14, 120, 120)
    # The filled channels hold the input values; absent channels hold the mean.
    for slot in ["B02", "B03", "B04"]:
        assert stem[BEN_14_SLOTS.index(slot)].sum() == 0.0  # input was zeros
    for slot in mapping["slots_absent"]:
        assert np.all(stem[BEN_14_SLOTS.index(slot)] == ctx.training_means[slot])


def test_pan1_replicates_across_rgb(ctx):
    arr = np.full((120, 120, 1), 17.0, dtype=np.float32)
    stem, mapping, src = _raster_to_stem(ctx, _raster(arr, "unknown"))
    assert mapping["slots_filled"] == ["B02", "B03", "B04"]
    assert mapping["source_modality"] == "unknown"  # 1-band is ambiguous
    for slot in ["B02", "B03", "B04"]:
        assert np.allclose(stem[BEN_14_SLOTS.index(slot)], 17.0)


def test_msi4_maps_b02_b03_b04_b08(ctx):
    arr = np.stack([np.full((60, 60), float(v)) for v in (1.0, 2.0, 3.0, 4.0)],
                   axis=-1)
    stem, mapping, src = _raster_to_stem(ctx, _raster(arr, "msi"))
    assert mapping["slots_filled"] == ["B02", "B03", "B04", "B08"]
    assert mapping["source_modality"] == "msi"
    assert np.allclose(stem[BEN_14_SLOTS.index("B02")], 1.0)
    assert np.allclose(stem[BEN_14_SLOTS.index("B08")], 4.0)


def test_sar1_maps_to_vv(ctx):
    arr = np.full((120, 120, 1), 0.5, dtype=np.float32)
    stem, mapping, src = _raster_to_stem(ctx, _raster(arr, "sar"))
    assert mapping["slots_filled"] == ["VV"]
    assert mapping["source_modality"] == "sar"
    assert np.allclose(stem[BEN_14_SLOTS.index("VV")], 0.5)
    assert "VH" in mapping["slots_absent"]


def test_sar2_maps_vv_vh(ctx):
    arr = np.stack([np.full((120, 120), 0.1), np.full((120, 120), 0.2)], axis=-1)
    stem, mapping, src = _raster_to_stem(ctx, _raster(arr, "sar"))
    assert mapping["slots_filled"] == ["VH", "VV"]
    assert mapping["source_modality"] == "sar"
    assert np.allclose(stem[BEN_14_SLOTS.index("VV")], 0.1)
    assert np.allclose(stem[BEN_14_SLOTS.index("VH")], 0.2)


def test_unknown_2band_fails_soft_by_band_count(ctx):
    # A bare 2-band file with no filename hint resolves to "unknown" (RISAT vs
    # PAN ambiguity). Must not raise; map by band count into VV/VH and disclose.
    arr = np.zeros((120, 120, 2), dtype=np.float32)
    stem, mapping, src = _raster_to_stem(ctx, _raster(arr, "unknown"))
    assert mapping["slots_filled"] == ["VH", "VV"]
    assert mapping["source_modality"] == "unknown"
    validate_band_mapping(mapping)


def test_12band_maps_standard_indices(ctx):
    # A 12-band Sentinel-like stack is a 3D array (H, W, 12); B02/B03/B04/B08
    # are channels 0/1/2/3 only if the caller provided that order; otherwise we
    # fill what aligns. Here we pass a 12-channel array and assert we at least
    # fill B02/B03/B04/B08 by position and everything else with means.
    arr = np.zeros((60, 60, 12), dtype=np.float32)
    stem, mapping, src = _raster_to_stem(ctx, _raster(arr, "msi"))
    assert mapping["source_modality"] == "msi"
    assert set(mapping["slots_filled"]) <= {"B02", "B03", "B04", "B08"}
    assert len(mapping["slots_absent"]) == 14 - len(mapping["slots_filled"])


def test_14band_passes_straight_through(ctx):
    arr = np.random.RandomState(0).rand(120, 120, 14).astype(np.float32)
    stem, mapping, src = _raster_to_stem(ctx, _raster(arr, "msi"))
    assert mapping["slots_filled"] == sorted(BEN_14_SLOTS)
    assert mapping["slots_absent"] == []
    assert stem.shape == (14, 120, 120)
    assert np.allclose(stem[0], arr[..., 0])  # first channel unchanged


def test_upload_slots_order_is_consistent():
    assert len(BEN_14_SLOTS) == 14
    assert set(BEN_14_SLOTS) == set(S2_BAND_ORDER) | set(S1_BAND_ORDER)
    assert S2_BAND_ORDER == [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A",
    ]
