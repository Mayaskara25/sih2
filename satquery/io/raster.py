"""The one and only place a raster is opened anywhere in this repo.

PLAN.md §4.4 (Image IO contract): every specialist, adapter, and UI path
loads pixels through `load_raster`. No other module may call
`rasterio.open` or `PIL.Image.open` directly - that duplication is exactly
what caused `old_files/models_registry.py` and `old_files/BT_CM.py` to carry
two diverging, buggy copies of `load_image` (PLAN.md §2.5).

The hidden ISRO evaluation set is 1-band Cartosat-2S panchromatic, 4-band
Cartosat MSI, and 1-2 band RISAT SAR (PLAN.md §2.3) - none of which are
3-band RGB. The old code's `src.read([1, 2, 3])` hard-crashes or silently
corrupts on all three. This module exists to get that right.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rasterio
from affine import Affine
from PIL import Image, UnidentifiedImageError

from satquery.io.modality import ModalityDecision, resolve_modality

# Extensions handled by each backend. Anything else falls through to a
# best-effort PIL attempt (PIL supports a long tail of formats), and if that
# fails too we raise RasterReadError with the real cause attached.
_RASTERIO_EXTS = (".tif", ".tiff")

# Sentinel-2 / BigEarthNet band index of B02 (Blue), B03 (Green), B04 (Red)
# within a 12-band (no B10, "L2A"-style) or 13-band (with B10, "L1C"-style)
# stack ordered B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, [B10,]
# B11, B12. B10 sits between B09 and B11 in the 13-band case, so it does not
# shift the B02/B03/B04 indices at all - they are 1/2/3 either way.
_S2_BLUE_IDX, _S2_GREEN_IDX, _S2_RED_IDX = 1, 2, 3

# Cartosat-2S MSI 4-band order assumed here: B02 (Blue), B03 (Green),
# B04 (Red), B08 (NIR) - i.e. Blue, Green, Red, NIR. This is the same
# assumption PLAN.md §4.5 requires `benclip`'s band-mapping policy to use
# ("Cartosat MSI (4-band) -> B02/B03/B04/B08 (blue/green/red/NIR)"), and is
# kept identical here on purpose so the two modules never disagree about
# what channel 0..3 mean for the same file. PLAN.md §4.5 flags this as an
# assumption to verify against a real Bhoonidhi product - it has not been
# independently verified here.
_CARTOSAT_BLUE_IDX, _CARTOSAT_GREEN_IDX, _CARTOSAT_RED_IDX, _CARTOSAT_NIR_IDX = 0, 1, 2, 3


class RasterReadError(RuntimeError):
    """The file exists but rasterio/PIL could not decode it as an image."""


@dataclass
class RasterInput:
    """Uniform in-memory representation of any raster this system reads.

    Fields are exactly the ones frozen by PLAN.md §4.4, plus
    `modality_decision`, which is required by the same section's rule that
    "whichever [tier] was used is recorded" so the controller can copy it
    into the execution trace (PLAN.md §4.2 `inputs[].modality`) without
    re-deriving it.
    """

    array: np.ndarray                 # (H, W, C), native dtype preserved
    modality: str                     # "optical" | "msi" | "sar" | "unknown"
    band_count: int
    band_names: List[str]
    crs: Optional[str]
    transform: Optional[Affine]
    is_georeferenced: bool
    source_path: str
    display_rgb: np.ndarray           # (H, W, 3) uint8, always safe to imshow
    modality_decision: ModalityDecision


# --------------------------------------------------------------------------
# Numeric helpers - shared by every band-count branch below. Each is written
# to be crash-proof against the specific failure modes PLAN.md §2.5 calls
# out: all-constant images (div by zero), NaN/inf pixels, and SAR's huge
# dynamic range.
# --------------------------------------------------------------------------

def _percentile_stretch(band: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """2-98 percentile stretch to uint8. Never divides by zero, never emits NaN."""
    band = np.asarray(band, dtype=np.float64)
    finite_mask = np.isfinite(band)
    if not finite_mask.any():
        return np.zeros(band.shape, dtype=np.uint8)

    finite_vals = band[finite_mask]
    lo, hi = np.percentile(finite_vals, (low, high))
    if hi <= lo:
        # Degenerate/near-constant image (e.g. all-zero or single-value
        # scene) - fall back to true min/max so we don't divide by zero.
        lo, hi = float(finite_vals.min()), float(finite_vals.max())

    if hi <= lo:
        # Truly constant image: nothing to stretch, return mid-gray rather
        # than dividing by zero.
        stretched = np.full(band.shape, 0.5, dtype=np.float64)
    else:
        stretched = (band - lo) / (hi - lo)

    stretched = np.where(finite_mask, stretched, 0.0)
    stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0)
    stretched = np.clip(stretched, 0.0, 1.0)
    return (stretched * 255.0).round().astype(np.uint8)


def _to_db(band: np.ndarray) -> np.ndarray:
    """10*log10(intensity) with a guard against zero/negative/non-finite input.

    SAR intensity commonly spans several orders of magnitude; a linear
    percentile stretch on that range collapses almost every pixel to black
    or white. PLAN.md §4.4 requires dB/log scaling before the stretch for
    any SAR modality - this must run BEFORE `_percentile_stretch`, never
    after.
    """
    band = np.asarray(band, dtype=np.float64)
    band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)

    positive = band[band > 0]
    if positive.size:
        # Floor well below the smallest observed positive value so real
        # signal is never clipped, but zero/negative pixels (invalid
        # returns, no-data) don't produce -inf.
        floor = max(float(positive.min()) * 1e-3, 1e-12)
    else:
        floor = 1e-12

    safe = np.clip(band, floor, None)
    return 10.0 * np.log10(safe)


def _replicate_to_three(band2d: np.ndarray, *, is_sar: bool) -> np.ndarray:
    if is_sar:
        band2d = _to_db(band2d)
    single = _percentile_stretch(band2d)
    return np.stack([single, single, single], axis=-1)


def _two_band_display(arr: np.ndarray, *, is_sar: bool) -> np.ndarray:
    """Synthesize a 3rd display channel for a 2-band raster.

    2 bands is the RISAT dual-pol (VV, VH) case (PLAN.md §4.4). Display as
    (VV, VH, mean(VV, VH)) so the composite is a genuine function of both
    polarizations rather than a duplicated/garbage channel. For SAR, each
    band is dB-scaled before stretching; for non-SAR 2-band rasters (not
    expected in practice, but must not crash) the same synthesis runs on
    the raw values.
    """
    b0, b1 = arr[..., 0], arr[..., 1]
    if is_sar:
        b0, b1 = _to_db(b0), _to_db(b1)
    third = (b0 + b1) / 2.0
    return np.stack(
        [_percentile_stretch(b0), _percentile_stretch(b1), _percentile_stretch(third)],
        axis=-1,
    )


def _passthrough_three_band_display(arr: np.ndarray, *, is_sar: bool) -> np.ndarray:
    channels = []
    for i in range(3):
        band = arr[..., i]
        if is_sar:
            band = _to_db(band)
        channels.append(_percentile_stretch(band))
    return np.stack(channels, axis=-1)


def _cartosat_msi_display(arr: np.ndarray) -> np.ndarray:
    """4-band Cartosat-2S MSI -> RGB using the Blue/Green/Red/NIR order
    documented at module scope (matches PLAN.md §4.5's `benclip` policy).
    """
    r = _percentile_stretch(arr[..., _CARTOSAT_RED_IDX])
    g = _percentile_stretch(arr[..., _CARTOSAT_GREEN_IDX])
    b = _percentile_stretch(arr[..., _CARTOSAT_BLUE_IDX])
    return np.stack([r, g, b], axis=-1)


def _quad_pol_sar_display(arr: np.ndarray) -> np.ndarray:
    """4-band SAR (e.g. RISAT-1 FRS-1 quad-pol HH/HV/VH/VV) -> dB-scaled RGB.

    A 4-band raster that resolves to modality "sar" is quad-pol, not an MSI
    stack - PLAN.md's band-count table only names 1-2 band SAR explicitly,
    but the modality decision is what must drive this branch, not the band
    count alone. Polarization order is not standardized across products, so
    this takes the first three bands (assumed HH, HV, VH) and dB-scales each
    independently before stretching, exactly like the 1- and 2-band SAR
    paths. Never fall through to `_cartosat_msi_display`'s plain linear
    stretch for a raster the resolver has called "sar".
    """
    return _passthrough_three_band_display(arr[..., :3], is_sar=True)


def _sentinel2_display(arr: np.ndarray, *, is_sar: bool = False) -> np.ndarray:
    """12-13 band optical/MSI -> RGB via B04/B03/B02 (see index comment above).

    `is_sar` exists only so a 12/13-band raster that the modality resolver
    calls "sar" (an unusual multi-band SAR stack) still gets dB scaling
    before the stretch instead of silently reverting to a linear stretch -
    PLAN.md §4.4 requires the SAR branch to be driven by resolved modality,
    not by band count.
    """
    r = arr[..., _S2_RED_IDX]
    g = arr[..., _S2_GREEN_IDX]
    b = arr[..., _S2_BLUE_IDX]
    if is_sar:
        r, g, b = _to_db(r), _to_db(g), _to_db(b)
    return np.stack([_percentile_stretch(r), _percentile_stretch(g), _percentile_stretch(b)], axis=-1)


def _fallback_display(arr: np.ndarray, *, is_sar: bool) -> np.ndarray:
    """Any band count we don't have a named policy for (5-11, 14+).

    PLAN.md §4.4: "do not crash; take the first 1 or 3 bands ... and record
    it." `arr` always has >= 5 channels by the time this is called (see
    `_build_display_rgb`), so we always have >= 3 to take.
    """
    return _passthrough_three_band_display(arr[..., :3], is_sar=is_sar)


def _build_display_rgb(arr: np.ndarray, band_count: int, *, is_sar: bool) -> np.ndarray:
    """Build the always-safe (H, W, 3) uint8 display image.

    `is_sar` is the single flag that drives the log-before-stretch branch
    everywhere below. It must reflect the resolved modality, never band
    count alone - a resolved-"sar" input must never reach a plain linear
    stretch, regardless of how many bands it has (PLAN.md §4.4).
    """
    if band_count == 1:
        return _replicate_to_three(arr[..., 0], is_sar=is_sar)
    if band_count == 2:
        return _two_band_display(arr, is_sar=is_sar)
    if band_count == 3:
        return _passthrough_three_band_display(arr, is_sar=is_sar)
    if band_count == 4:
        if is_sar:
            return _quad_pol_sar_display(arr)
        return _cartosat_msi_display(arr)
    if band_count in (12, 13):
        return _sentinel2_display(arr, is_sar=is_sar)
    return _fallback_display(arr, is_sar=is_sar)


def _is_expected_band_count(band_count: int) -> bool:
    return band_count in (1, 2, 3, 4, 12, 13)


# --------------------------------------------------------------------------
# Backend loaders
# --------------------------------------------------------------------------

def _load_via_rasterio(path: str):
    try:
        src = rasterio.open(path)
    except Exception as exc:  # rasterio raises its own RasterioIOError subclasses
        raise RasterReadError(f"Could not open raster {path!r} with rasterio: {exc}") from exc

    with src:
        try:
            raw = src.read()  # (bands, H, W), native dtype
        except Exception as exc:
            raise RasterReadError(f"Could not read pixel data from {path!r}: {exc}") from exc

        arr = np.transpose(raw, (1, 2, 0))  # (H, W, bands)
        band_count = arr.shape[2]

        crs = src.crs
        transform = src.transform
        is_georeferenced = crs is not None
        crs_str = crs.to_string() if crs is not None else None
        # `is_georeferenced` (crs presence) and "has a usable pixel
        # transform" are different questions: some SAR ground-range
        # products (e.g. RISAT GRD) carry real pixel spacing with no map
        # projection. Keep a non-identity transform even without a CRS so
        # W4 (registration) and W5 (fusion) still have pixel spacing to
        # work with; only drop it when rasterio reports the untransformed
        # default (no georeferencing info at all was set on the file).
        transform_out = transform if transform is not None and transform != Affine.identity() else None

        descriptions = src.descriptions or ()
        band_names = [
            (name if name else f"Band_{i + 1}")
            for i, name in enumerate(
                list(descriptions) + [None] * (band_count - len(descriptions))
            )
        ]

        try:
            metadata_tags = dict(src.tags())
        except Exception:
            metadata_tags = {}

    return arr, band_count, crs_str, transform_out, is_georeferenced, band_names, metadata_tags


def _load_via_pil(path: str):
    try:
        img = Image.open(path)
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise RasterReadError(f"Could not open image {path!r} with PIL: {exc}") from exc

    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")

    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    band_count = arr.shape[2]

    band_names = ["Band_1"] if band_count == 1 else ["R", "G", "B"][:band_count]
    if len(band_names) < band_count:
        band_names += [f"Band_{i + 1}" for i in range(len(band_names), band_count)]

    # PNG/JPEG carry no CRS and, for our purposes, no reliable sensor
    # metadata tags - PLAN.md §4.4 requires is_georeferenced=False, crs=None,
    # transform=None for these formats.
    return arr, band_count, None, None, False, band_names, {}


def load_raster(path: str, modality: Optional[str] = None) -> RasterInput:
    """Load any raster this system can encounter into a `RasterInput`.

    Args:
        path: path to a .tif/.tiff (opened via rasterio) or any PIL-readable
            image such as .png/.jpg (opened via PIL). Every other format is
            attempted via PIL as a best-effort fallback.
        modality: optional explicit modality override (tier 1 in
            PLAN.md §4.4's precedence). One of "optical" | "msi" | "sar" |
            "unknown". If omitted, modality is resolved from filename,
            metadata, and band-count heuristics - see `satquery.io.modality`.

    Returns:
        A populated RasterInput. `display_rgb` is always (H, W, 3) uint8,
        always finite, and always safe to hand to `PIL.Image.fromarray` or
        `matplotlib.pyplot.imshow` regardless of the input's band count,
        dtype, or dynamic range.

    Raises:
        FileNotFoundError: `path` does not exist.
        RasterReadError: the file exists but could not be decoded.
        ValueError: `modality` is supplied but not a valid modality string.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raster file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in _RASTERIO_EXTS:
        arr, band_count, crs_str, transform, is_georef, band_names, metadata_tags = _load_via_rasterio(path)
    else:
        try:
            arr, band_count, crs_str, transform, is_georef, band_names, metadata_tags = _load_via_pil(path)
        except RasterReadError:
            # Last resort: some GeoTIFF-like files show up with an
            # unexpected extension. Try rasterio before giving up.
            try:
                arr, band_count, crs_str, transform, is_georef, band_names, metadata_tags = _load_via_rasterio(path)
            except RasterReadError:
                raise

    decision = resolve_modality(
        user_modality=modality,
        source_path=path,
        band_count=band_count,
        metadata_tags=metadata_tags,
    )
    resolved_modality = decision.modality
    # Drives the display path's log-before-stretch branch. Computed from the
    # tier decision BEFORE the "anomalous band count -> unknown" override
    # below, so a SAR scene with an unexpected band count still gets dB
    # scaling for display even though it is honestly *reported* as
    # "unknown" - the honesty requirement is about what we claim to know,
    # not about degrading the one thing (log scaling) we're confident about.
    is_sar_for_display = resolved_modality == "sar"

    if not _is_expected_band_count(band_count) and decision.mechanism != "user":
        # PLAN.md §4.4: unexpected band counts must not crash and must not
        # be guessed at - forced to "unknown" unless the user explicitly
        # overrode the modality (tier 1 is never second-guessed).
        decision = ModalityDecision(
            modality="unknown",
            mechanism=decision.mechanism,
            reason=(
                f"band count {band_count} is outside the handled set "
                f"(1, 2, 3, 4, 12, 13); forcing modality to 'unknown' "
                f"regardless of the {decision.mechanism} tier's guess "
                f"({decision.modality}). Original reason: {decision.reason}"
            ),
        )
        resolved_modality = "unknown"

    display_rgb = _build_display_rgb(arr, band_count, is_sar=is_sar_for_display)

    return RasterInput(
        array=arr,
        modality=resolved_modality,
        band_count=band_count,
        band_names=band_names,
        crs=crs_str,
        transform=transform,
        is_georeferenced=is_georef,
        source_path=path,
        display_rgb=display_rgb,
        modality_decision=decision,
    )
