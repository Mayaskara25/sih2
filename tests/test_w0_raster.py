"""Tests for satquery.io.raster - the only place a raster is opened.

All fixtures are synthetic, generated on the fly with rasterio/PIL in
tmp_path. Nothing here depends on data/ or old_files/imgs/.
"""

import numpy as np
import pytest
import rasterio
from affine import Affine
from PIL import Image

from satquery.io.raster import (
    RasterReadError,
    _percentile_stretch,
    _to_db,
    load_raster,
)

_DEFAULT_TRANSFORM = Affine.translation(10.0, 20.0) @ Affine.scale(0.5, -0.5)


def _write_geotiff(path, array, crs="EPSG:4326", transform=_DEFAULT_TRANSFORM):
    """array: (H, W) or (H, W, C)."""
    if array.ndim == 2:
        array = array[:, :, None]
    height, width, count = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for i in range(count):
            dst.write(array[:, :, i], i + 1)
    return path


def _random_band(shape, dtype=np.uint16, low=0, high=4000, seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        return rng.integers(low, high, size=shape).astype(dtype)
    return rng.uniform(low, high, size=shape).astype(dtype)


def _hist_entropy(img):
    hist, _ = np.histogram(img, bins=256, range=(0, 255))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _wide_dynamic_range_sar_intensity(shape, seed):
    """Realistic SAR speckle: low-backscatter background plus a sparse
    population of strong scatterers 2-4 orders of magnitude brighter -
    exactly the scene PLAN.md §4.4 calls out as unreadable under a plain
    linear stretch.
    """
    rng = np.random.default_rng(seed)
    background = rng.exponential(scale=1.0, size=shape) * 2.0 + 0.01
    intensity = background.copy()
    bright_mask = rng.random(shape) < 0.08
    bright = rng.exponential(scale=1.0, size=shape) * 20000.0 + 500.0
    intensity[bright_mask] = bright[bright_mask]
    return intensity.astype(np.float32)


# ---------------------------------------------------------------------
# Band-count handling
# ---------------------------------------------------------------------

@pytest.mark.parametrize("band_count", [1, 2, 3, 4, 12])
def test_load_raster_band_counts(tmp_path, band_count):
    h, w = 16, 20
    arr = np.stack([_random_band((h, w), seed=i) for i in range(band_count)], axis=-1)
    path = str(tmp_path / f"scene_{band_count}band.tif")
    _write_geotiff(path, arr)

    result = load_raster(path)

    assert result.band_count == band_count
    assert result.array.shape == (h, w, band_count)
    assert result.display_rgb.shape == (h, w, 3)
    assert result.display_rgb.dtype == np.uint8
    assert result.is_georeferenced is True
    assert result.crs is not None
    assert result.transform is not None
    assert result.source_path == path
    assert len(result.band_names) == band_count
    assert result.modality_decision is not None


def test_13_band_sentinel2_like(tmp_path):
    h, w = 10, 10
    arr = np.stack([_random_band((h, w), seed=i) for i in range(13)], axis=-1)
    path = str(tmp_path / "s2_tile.tif")
    _write_geotiff(path, arr)

    result = load_raster(path, modality="msi")

    assert result.band_count == 13
    assert result.display_rgb.shape == (h, w, 3)


def test_msi_display_selects_documented_rgb_bands(tmp_path):
    # Pins the exact band indices used for the 12/13-band Sentinel-2 display
    # subset, so a reordering of _S2_RED_IDX/_S2_GREEN_IDX/_S2_BLUE_IDX (or
    # an accidental log-scaling step creeping into the optical path) fails
    # loudly instead of silently producing a wrong false-color image that
    # every shape/dtype-only test above would still pass.
    h, w = 16, 16
    arr = np.stack([_random_band((h, w), seed=i) for i in range(12)], axis=-1)
    path = str(tmp_path / "s2_indexcheck.tif")
    _write_geotiff(path, arr)

    result = load_raster(path)

    assert np.array_equal(result.display_rgb[..., 0], _percentile_stretch(result.array[..., 3]))  # B04 red
    assert np.array_equal(result.display_rgb[..., 1], _percentile_stretch(result.array[..., 2]))  # B03 green
    assert np.array_equal(result.display_rgb[..., 2], _percentile_stretch(result.array[..., 1]))  # B02 blue


def test_cartosat_msi_display_selects_documented_rgb_bands(tmp_path):
    # Same pin for the 4-band Cartosat-2S MSI case: assumed order is
    # Blue, Green, Red, NIR (indices 0,1,2,3) per PLAN.md §4.5.
    h, w = 16, 16
    arr = np.stack([_random_band((h, w), seed=i) for i in range(4)], axis=-1)
    path = str(tmp_path / "cartosat_msi_indexcheck.tif")
    _write_geotiff(path, arr)

    result = load_raster(path, modality="msi")

    assert np.array_equal(result.display_rgb[..., 0], _percentile_stretch(result.array[..., 2]))  # red
    assert np.array_equal(result.display_rgb[..., 1], _percentile_stretch(result.array[..., 1]))  # green
    assert np.array_equal(result.display_rgb[..., 2], _percentile_stretch(result.array[..., 0]))  # blue


def test_unexpected_band_count_forced_unknown(tmp_path):
    h, w = 8, 8
    arr = np.stack([_random_band((h, w), seed=i) for i in range(7)], axis=-1)
    path = str(tmp_path / "weird_scene.tif")
    _write_geotiff(path, arr)

    result = load_raster(path)

    assert result.band_count == 7
    assert result.modality == "unknown"
    assert result.display_rgb.shape == (h, w, 3)
    assert result.display_rgb.dtype == np.uint8


def test_unexpected_band_count_user_override_respected(tmp_path):
    h, w = 8, 8
    arr = np.stack([_random_band((h, w), seed=i) for i in range(7)], axis=-1)
    path = str(tmp_path / "weird_scene2.tif")
    _write_geotiff(path, arr)

    result = load_raster(path, modality="sar")

    # Explicit tier-1 override must never be second-guessed, even for an
    # unusual band count.
    assert result.modality == "sar"
    assert result.modality_decision.mechanism == "user"


# ---------------------------------------------------------------------
# PNG/JPEG round trip (no CRS)
# ---------------------------------------------------------------------

def test_png_round_trip(tmp_path):
    h, w = 12, 15
    arr = (np.random.default_rng(1).uniform(0, 255, size=(h, w, 3))).astype(np.uint8)
    path = str(tmp_path / "benchmark.png")
    Image.fromarray(arr, mode="RGB").save(path)

    result = load_raster(path)

    assert result.band_count == 3
    assert result.is_georeferenced is False
    assert result.crs is None
    assert result.transform is None
    assert result.display_rgb.shape == (h, w, 3)
    assert result.display_rgb.dtype == np.uint8


def test_grayscale_png_round_trip(tmp_path):
    h, w = 12, 15
    arr = (np.random.default_rng(2).uniform(0, 255, size=(h, w))).astype(np.uint8)
    path = str(tmp_path / "pan.png")
    Image.fromarray(arr, mode="L").save(path)

    result = load_raster(path)

    assert result.band_count == 1
    assert result.is_georeferenced is False
    assert result.display_rgb.shape == (h, w, 3)


# ---------------------------------------------------------------------
# SAR dB scaling vs. naive linear stretch
# ---------------------------------------------------------------------

def test_sar_db_scaling_has_materially_better_dynamic_range(tmp_path):
    h, w = 64, 64
    intensity = _wide_dynamic_range_sar_intensity((h, w), seed=42)

    path = str(tmp_path / "risat_sar_scene.tif")
    _write_geotiff(path, intensity, crs=None)

    result = load_raster(path, modality="sar")
    db_display = result.display_rgb[..., 0]

    naive_display = _percentile_stretch(intensity)

    db_entropy = _hist_entropy(db_display)
    naive_entropy = _hist_entropy(naive_display)

    # The naive linear stretch of raw SAR intensity collapses almost the
    # entire background into the bottom histogram bin because a handful of
    # bright scatterers dominate the 2-98 percentile range; the dB-scaled
    # version spreads pixel values across the full 0-255 range instead.
    # Histogram entropy (bits) captures that spread directly.
    assert db_entropy > naive_entropy * 2

    naive_bottom_frac = float(np.mean(naive_display <= 5))
    db_bottom_frac = float(np.mean(db_display <= 5))
    assert naive_bottom_frac > 0.5  # naive: most of the scene crushed to near-black
    assert db_bottom_frac < naive_bottom_frac / 2  # dB: materially better spread

    db_std = float(np.std(db_display.astype(np.float64)))
    naive_std = float(np.std(naive_display.astype(np.float64)))
    assert db_std > naive_std


def test_quad_pol_sar_4band_uses_db_scaling_not_linear_msi_path(tmp_path):
    # A 4-band raster that resolves to "sar" (e.g. RISAT-1 FRS-1 quad-pol
    # HH/HV/VH/VV) must NOT fall through to the Cartosat-MSI linear-stretch
    # display path just because it happens to have 4 bands - the branch is
    # driven by resolved modality, not band count (PLAN.md §4.4).
    h, w = 48, 48
    band0 = _wide_dynamic_range_sar_intensity((h, w), seed=100)
    band1 = _wide_dynamic_range_sar_intensity((h, w), seed=101)
    band2 = _wide_dynamic_range_sar_intensity((h, w), seed=102)
    band3 = _wide_dynamic_range_sar_intensity((h, w), seed=103)
    arr = np.stack([band0, band1, band2, band3], axis=-1)
    path = str(tmp_path / "risat_quadpol.tif")
    _write_geotiff(path, arr, crs=None)

    result = load_raster(path, modality="sar")
    assert result.band_count == 4
    assert result.modality == "sar"

    naive_display = _percentile_stretch(band0)
    db_channel = result.display_rgb[..., 0]

    assert _hist_entropy(db_channel) > _hist_entropy(naive_display) * 1.5


def test_anomalous_band_count_sar_display_still_db_scaled(tmp_path):
    # A 5-band raster is outside the handled set, so the REPORTED modality
    # is honestly forced to "unknown" (see test_unexpected_band_count_forced_unknown
    # below) - but the resolver's underlying SAR guess (from the filename)
    # must still drive the display's log-before-stretch branch. Reporting
    # "unknown" is about honesty in what we claim to know; it must not
    # silently degrade the one thing (log scaling) the resolver was
    # confident about, or a genuinely-SAR scene with an odd band count
    # renders unreadable.
    h, w = 32, 32
    bands = [_wide_dynamic_range_sar_intensity((h, w), seed=200 + i) for i in range(5)]
    arr = np.stack(bands, axis=-1)
    path = str(tmp_path / "risat_sar_5band.tif")
    _write_geotiff(path, arr, crs=None)

    result = load_raster(path)  # no override - filename "sar" hint must fire

    assert result.modality == "unknown"  # honest report: 5 bands isn't a known SAR shape
    assert result.modality_decision.mechanism == "filename"

    naive_display = _percentile_stretch(bands[0])
    db_channel = result.display_rgb[..., 0]
    assert _hist_entropy(db_channel) > _hist_entropy(naive_display) * 1.5


def test_to_db_guards_zero_and_negative():
    band = np.array([[0.0, -5.0], [1.0, 100.0]], dtype=np.float64)
    db = _to_db(band)
    assert np.all(np.isfinite(db))


def test_sar_two_band_display_uses_both_channels(tmp_path):
    h, w = 20, 20
    vv = _random_band((h, w), dtype=np.float32, low=1, high=5000, seed=3)
    vh = _random_band((h, w), dtype=np.float32, low=1, high=2000, seed=4)
    arr = np.stack([vv, vh], axis=-1)
    path = str(tmp_path / "risat_dualpol.tif")
    _write_geotiff(path, arr)

    result = load_raster(path, modality="sar")

    assert result.band_count == 2
    assert result.display_rgb.shape == (h, w, 3)
    # The three display channels should not be identical to each other -
    # a real synthesis of VV/VH/combination, not a replicated garbage band.
    r, g, b = (result.display_rgb[..., i] for i in range(3))
    assert not np.array_equal(r, g)
    assert not np.array_equal(g, b)


# ---------------------------------------------------------------------
# Robustness: constant image, NaN/inf, missing file, corrupt file
# ---------------------------------------------------------------------

def test_all_constant_image_no_crash_no_nan(tmp_path):
    h, w = 10, 10
    arr = np.full((h, w), 42.0, dtype=np.float32)
    path = str(tmp_path / "constant.tif")
    _write_geotiff(path, arr)

    result = load_raster(path)

    assert result.display_rgb.shape == (h, w, 3)
    assert not np.isnan(result.display_rgb.astype(np.float64)).any()
    assert np.isfinite(result.display_rgb.astype(np.float64)).all()


def test_nan_inf_pixels_handled(tmp_path):
    h, w = 10, 10
    arr = np.random.default_rng(5).uniform(0, 100, size=(h, w)).astype(np.float32)
    arr[0, 0] = np.nan
    arr[1, 1] = np.inf
    arr[2, 2] = -np.inf
    path = str(tmp_path / "nan_inf.tif")
    _write_geotiff(path, arr)

    result = load_raster(path)

    assert not np.isnan(result.display_rgb.astype(np.float64)).any()
    assert np.isfinite(result.display_rgb.astype(np.float64)).all()
    # display_rgb is uint8, so isnan/isfinite above are trivially satisfied
    # even if the NaN/inf pixels had poisoned the percentile stretch and
    # collapsed the whole image to black. Assert the stretch actually
    # preserved the real dynamic range of the other 97 finite pixels.
    assert np.unique(result.display_rgb).size > 10


def test_missing_file_raises_file_not_found_error():
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        load_raster("/tmp/does_not_exist_satquery_w0_test.tif")


def test_corrupt_file_raises_clear_exception(tmp_path):
    path = tmp_path / "corrupt.tif"
    path.write_bytes(b"this is not a valid tiff or image file at all")

    with pytest.raises(RasterReadError):
        load_raster(str(path))


# ---------------------------------------------------------------------
# Modality plumbing end-to-end through load_raster
# ---------------------------------------------------------------------

def test_modality_resolved_from_filename_when_no_override(tmp_path):
    h, w = 8, 8
    arr = _random_band((h, w), seed=9)
    path = str(tmp_path / "RISAT_GRD_VV.tif")
    _write_geotiff(path, arr)

    result = load_raster(path)

    assert result.modality == "sar"
    assert result.modality_decision.mechanism == "filename"


def test_modality_user_override_wins_over_filename(tmp_path):
    h, w = 8, 8
    arr = _random_band((h, w), seed=10)
    path = str(tmp_path / "RISAT_GRD_VV.tif")
    _write_geotiff(path, arr)

    result = load_raster(path, modality="optical")

    assert result.modality == "optical"
    assert result.modality_decision.mechanism == "user"


def test_percentile_stretch_constant_band_no_div_by_zero():
    band = np.zeros((5, 5), dtype=np.float32)
    stretched = _percentile_stretch(band)
    assert stretched.dtype == np.uint8
    assert np.isfinite(stretched.astype(np.float64)).all()
