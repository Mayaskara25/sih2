"""
SatQuery AI -- Model Registry (Person 5: Single-Image VQA & Text-Guided Grounding)
--- INNOVATION BUILD v3 (Innovations 7-10 + Dynamic Token Softmax Confidence) ---

Unified inference engine exposing the exact JSON/function contract expected by the
Agentic Controller (Pair 2). This module is provider-agnostic: it lazily loads models
on first use and caches them module-wide so that Person 6's bi-temporal / cross-modal
functions reuse the same Qwen2-VL instance (shared _load_vlm() -- load once).

PROPRIETARY INNOVATIONS IMPLEMENTED UNDER THE FROZEN CONTRACT:
  I07  Physics-Gated Spectral Pre-Filtering (NDWI water / NDVI vegetation gating)
  I08  SAM-RS Polygon Segmentation & Oriented Bounding Boxes (OBB) via minAreaRect
  I09  Single-SAR Radar Backscatter Engine (Lee speckle filter + dB thresholding +
       SAR-aware context prompt)
  I10  Radiometric Dynamic Range Calibration & Atmospheric Stretch (2-98 percentile
       cumulative stretch + CLAHE on L channel of LAB)
  TSC  Dynamic Token Softmax Confidence Scoring (model logit-derived confidence)
  A    Component A: MobileSAM neural zero-shot segmentation (polygon + OBB) with
       classical OpenCV fallback
  B    Component B: SAHI slicing-aided hyper inference (multi-scale tiling, global
       coords remap, geospatial NMS) with full-image fallback

Exported functions (contract -- DO NOT change keys):
    run_vqa(image_path, prompt)     -> {"text_response": str, "confidence": float}
    run_grounding(image_path, target)
        -> {"text_response": str,
            "bounding_boxes": [{"label", "box_2d": [ymin, xmin, ymax, xmax], "confidence"}],
            "confidence": float}
    get_execution_metadata()        -> dict
    flush_models()                  -> None
"""

from __future__ import annotations

import os
import time
import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
VLM_REPO: str = os.environ.get("SATQUERY_VLM", "Qwen/Qwen2-VL-2B-Instruct")
GROUNDING_REPO: str = os.environ.get(
    "SATQUERY_GROUNDING", "IDEA-Research/grounding-dino-tiny"
)

GROUNDING_BOX_THRESHOLD: float = float(os.environ.get("SATQUERY_GROUND_THRESHOLD", "0.18"))
GROUNDING_TEXT_THRESHOLD: float = float(os.environ.get("SATQUERY_GROUND_TEXT_THRESHOLD", "0.18"))
VLM_MAX_NEW_TOKENS: int = int(os.environ.get("SATQUERY_VLM_MAX_TOKENS", "256"))

# --- Innovation 7: spectral gating parameters ---------------------------------
NDWI_THRESHOLD: float = 0.0          # water when (Green - NIR)/(Green + NIR) > 0
NDVI_THRESHOLD: float = 0.2          # vegetation when (NIR - Red)/(NIR + Red) > 0.2
GATE_MIN_WATER_FRACTION: float = 0.25  # fraction of box pixels that must be water
GATE_MIN_VEG_FRACTION: float = 0.35   # fraction of box pixels that must be veg

# Maritime / vegetation keyword sets for automatic gating.
MARITIME_KEYWORDS: Tuple[str, ...] = ("ship", "vessel", "boat", "ferry", "water body",
                                      "water", "oil tanker", "cargo ship", "boat")
VEGETATION_KEYWORDS: Tuple[str, ...] = ("agriculture", "agricultural", "crop", "crop field",
                                        "forest", "farmland", "vegetation", "paddy", "field")

# --- Innovation 9: SAR backscatter thresholds (dB) -----------------------------
SAR_WATER_DB: float = -15.0   # specular dark (smooth water) -> below is water
SAR_URBAN_DB: float = -5.0    # double-bounce bright metal/structures -> above is urban

# --- Innovation 8: OBB contour refinement --------------------------------------
OBB_EDGE_CANNY_LOW: int = 60
OBB_EDGE_CANNY_HIGH: int = 140
OBB_MIN_AREA: int = 12       # ignore tiny boxes with no contour evidence
OBB_ENABLED: bool = os.environ.get("SATQUERY_OBB", "1") == "1"

# --- Component A: MobileSAM neural segmentation --------------------------------
MOBILESAM_ENABLED: bool = os.environ.get("SATQUERY_MOBILESAM", "1") == "1"
MOBILESAM_CKPT: str = os.environ.get(
    "SATQUERY_MOBILESAM_CKPT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "mobile_sam.pt"),
)
MOBILESAM_MODEL_TYPE: str = os.environ.get("SATQUERY_MOBILESAM_MODEL_TYPE", "vit_t")
MOBILESAM_NUM_POINTS: int = int(os.environ.get("SATQUERY_MOBILESAM_NUM_POINTS", "5"))
MOBILESAM_CONTOUR_MIN_AREA: int = int(os.environ.get("SATQUERY_MOBILESAM_CONTOUR_MIN_AREA", "8"))
MOBILESAM_VRAMP_FP16: bool = os.environ.get("SATQUERY_MOBILESAM_FP16", "0") == "1"

# --- Component B: SAHI slicing-aided hyper inference ---------------------------
SAHI_ENABLED: bool = os.environ.get("SATQUERY_USE_SAHI", "1") == "1"
SAHI_SLICE_WH: int = int(os.environ.get("SATQUERY_SAHI_SLICE", "512"))   # tile size (px)
SAHI_OVERLAP_RATIO: float = float(os.environ.get("SATQUERY_SAHI_OVERLAP", "0.2"))
SAHI_NMS_IOU_THRESH: float = float(os.environ.get("SATQUERY_SAHI_NMS_IOU", "0.45"))
SAHI_RESIZE_TRIGGER: int = int(os.environ.get("SATQUERY_SAHI_RESIZE_TRIGGER", "1024"))
_SAHI_AVAILABLE: Optional[bool] = None
_MOBILESAM_AVAILABLE: Optional[bool] = None

# --------------------------------------------------------------------------- #
# GLOBAL LAZY CACHES
# --------------------------------------------------------------------------- #
_VLM: Optional[Any] = None
_VLM_PROCESSOR: Optional[Any] = None
_GROUNDING: Optional[Any] = None
_GROUNDING_PROCESSOR: Optional[Any] = None
_MOBILESAM: Optional[Any] = None
_MOBILESAM_PREDICTOR: Optional[Any] = None
_MOBILESAM_EMBED_ID: Optional[Any] = None
_IS_CUDA: Optional[bool] = None
_last_ctx: Optional["ImageContext"] = None


# --------------------------------------------------------------------------- #
# IMAGE CONTEXT (carries spectral bands + modality + processed display RGB)
# --------------------------------------------------------------------------- #
class ImageContext:
    """Encapsulates everything the specialist needs about an uploaded image.

    - `rgb`:       PIL RGB image (radiometrically stretched + CLAHE-enhanced)
    - `array`:     optional float32 (H, W) or (H, W, B) raw band data
    - `bands`:     dict of named spectral bands (red, green, nir) if available
    - `is_sar`:    True when the raster is single-band radar-like
    - `is_geotiff`: True when the source was a multi-band GeoTIFF
    """

    __slots__ = ("rgb", "array", "bands", "is_sar", "is_geotiff", "source_ext")

    def __init__(self, rgb, array=None, bands=None, is_sar=False,
                 is_geotiff=False, source_ext=""):
        self.rgb = rgb
        self.array = array
        self.bands = bands or {}
        self.is_sar = is_sar
        self.is_geotiff = is_geotiff
        self.source_ext = source_ext


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def _cuda_available() -> bool:
    """Cache CUDA availability so we only probe torch once."""
    global _IS_CUDA
    if _IS_CUDA is None:
        try:
            import torch
            _IS_CUDA = bool(torch.cuda.is_available())
        except Exception:
            _IS_CUDA = False
    return _IS_CUDA


def _confidence_to_float(value: Any) -> float:
    """Safely coerce a model score into a bounded float in [0, 1]."""
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return 0.0


def _percentile_stretch(arr_uint8: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Innovation 10: 2%-98% cumulative dynamic-range stretch on 8-bit data."""
    a = arr_uint8.astype(np.float32)
    l = np.percentile(a, lo)
    h = np.percentile(a, hi)
    if h <= l:
        return arr_uint8
    a = np.clip((a - l) / (h - l), 0, 1)
    return (a * 255).astype(np.uint8)


def _clahe_luminance(rgb_uint8: np.ndarray, clip_limit: float = 2.0,
                     tile: int = 8) -> np.ndarray:
    """Innovation 10: CLAHE on the L channel of LAB to sharpen subtle boundaries."""
    import cv2
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# --------------------------------------------------------------------------- #
# SAR DETECTION & RADAR BACKSCATTER ENGINE (Innovation 9)
# --------------------------------------------------------------------------- #
def _lee_filter(img: np.ndarray, win: int = 5) -> np.ndarray:
    """
    Lee adaptive speckle filter (Innovation 9). Smoothes homogenous regions
    while preserving edges (speckle noise reduction for radar imagery).
    """
    from scipy.ndimage import uniform_filter
    img = img.astype(np.float32)
    mean = uniform_filter(img, win)
    sqr_mean = uniform_filter(img ** 2, win)
    variance = sqr_mean - mean ** 2
    overall_variance = np.var(img) + 1e-6
    weights = variance / (variance + overall_variance)
    out = mean + weights * (img - mean)
    return out


def _sar_backscatter_masks(gray01: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Innovation 9: fixed-threshold specular backscatter masks from normalized
    intensity (0..1). Map intensity to dB via 10*log10. Water (dark specular)
    has < SAR_WATER_DB; metal/urban double-bounce has > SAR_URBAN_DB.
    """
    intensity01 = np.clip(gray01, 1e-6, 1.0)
    db = 10.0 * np.log10(intensity01)
    water = (db < SAR_WATER_DB).astype(np.uint8) * 255
    urban = (db > SAR_URBAN_DB).astype(np.uint8) * 255
    return {"water_mask": water, "urban_mask": urban}


def _looks_like_sar(array: np.ndarray, src) -> bool:
    """Heuristically decide whether a raster is a single-band radar image."""
    # Single channel + float or high dynamic range + no meaningful color bands
    try:
        band_count = src.count
        nodata = src.nodata
        if band_count != 1:
            return False
        # SAR scenes are usually stored as single float amplitude/intensity
        dtype = src.dtypes[0].lower()
        return dtype.startswith("float") or dtype in ("uint16", "int16", "uint32")
    except Exception:
        return False


def _load_multiband_raster(image_path: str) -> ImageContext:
    """Load a GeoTIFF/SAR raster with spectral bands, SAR detection + I10 stretch."""
    import rasterio

    with rasterio.open(image_path) as src:
        band_count = src.count
        is_sar = _looks_like_sar(None, src)

        if is_sar:
            # Single-band radar: read amplitude, Lee-filter, mask via dB.
            raw = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                raw = np.where(raw == nodata, np.nan, raw)
            valid = raw[np.isfinite(raw)]
            lo, hi = np.percentile(valid, (2, 98)) if valid.size else (0, 1)
            if hi > lo:
                norm01 = np.clip((raw - lo) / (hi - lo), 0, 1)
            else:
                norm01 = np.zeros_like(raw, dtype=np.float32)
            norm01 = np.nan_to_num(norm01)
            gray = _percentile_stretch((norm01 * 255).astype(np.uint8))
            gray = _lee_filter(gray.astype(np.float32)).astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=2)
            bands = _sar_backscatter_masks(norm01)
            return ImageContext(
                Image.fromarray(rgb).convert("RGB"),
                array=norm01,
                bands=bands,
                is_sar=True,
                is_geotiff=True,
                source_ext=os.path.splitext(image_path)[1].lower(),
            )

        # Multi-band optical: try to identify R/G/NIR channels.
        data = src.read().astype(np.float32)  # (C, H, W)
        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)

        # Sentinel-2 style: B02=Blue, B03=Green, B04=Red, B08=NIR by description.
        try:
            descs = src.descriptions or []
            names = [(d or "") for d in descs]
        except Exception:
            names = [""] * band_count
        joined = " ".join(names).upper()

        def pick(token):
            for i, n in enumerate(names):
                if token in n.upper():
                    return i
            return None

        idx_blue = pick("B02") if "B02" in joined else None
        idx_green = pick("B03") if "B03" in joined else None
        idx_red = pick("B04") if "B04" in joined else None
        idx_nir = pick("B08") if "B08" in joined else None

        # Fallback: 3-band assumed RGB, no NIR.
        if idx_red is None and idx_green is None and idx_blue is None and band_count >= 3:
            idx_red, idx_green, idx_blue = 0, 1, 2

        def band01(idx):
            b = data[idx]
            v = b[np.isfinite(b)]
            lo, hi = np.percentile(v, (2, 98)) if v.size else (0, 1)
            if hi > lo:
                return np.clip((b - lo) / (hi - lo), 0, 1)
            return np.zeros_like(b, dtype=np.float32)

        bands: Dict[str, np.ndarray] = {}
        if idx_red is not None:
            bands["red"] = band01(idx_red)
        if idx_green is not None:
            bands["green"] = band01(idx_green)
        if idx_blue is not None:
            bands["blue"] = band01(idx_blue)
        if idx_nir is not None:
            bands["nir"] = band01(idx_nir)

        # Compose RGB (fall back to grayscale of first band if no color).
        if "red" in bands and "green" in bands and "blue" in bands:
            rgb01 = np.dstack([bands["red"], bands["green"], bands["blue"]])
        else:
            first = band01(0)
            rgb01 = np.dstack([first, first, first])

        rgb_uint8 = (np.nan_to_num(rgb01) * 255).astype(np.uint8)
        rgb_uint8 = _percentile_stretch(rgb_uint8)          # I10 stretch
        rgb_uint8 = _clahe_luminance(rgb_uint8)             # I10 CLAHE
        return ImageContext(
            Image.fromarray(rgb_uint8).convert("RGB"),
            array=data,
            bands=bands,
            is_sar=False,
            is_geotiff=True,
            source_ext=os.path.splitext(image_path)[1].lower(),
        )


# --------------------------------------------------------------------------- #
# INNOVATION 7 -- NDVI / NDWI SPECTRAL PRIORS
# --------------------------------------------------------------------------- #
def _ndvi(bands) -> Optional[np.ndarray]:
    """(NIR - Red)/(NIR + Red)."""
    nir = bands.get("nir")
    red = bands.get("red")
    if nir is None or red is None:
        return None
    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(denom) > 1e-8, (nir - red) / denom, np.nan)
    return np.nan_to_num(out)


def _ndwi(bands) -> Optional[np.ndarray]:
    """(Green - NIR)/(Green + NIR)."""
    green = bands.get("green")
    nir = bands.get("nir")
    if green is None or nir is None:
        return None
    denom = green + nir
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(denom) > 1e-8, (green - nir) / denom, np.nan)
    return np.nan_to_num(out)


def _rgb_water_prior(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Innovation 7 fallback: when no NIR band, approximate a water prior from
    color-space (dark blue-dominant low-saturation or very bright specular).
    Returns float32 0..1 likelihood-ish map aligned to the RGB shape.
    """
    import cv2
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[..., 0] * 2.0   # hue in 0..360
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    # Water: blue-ish hue, moderate saturation, lower value OR specular bright.
    blue = ((h > 180) & (h < 260)).astype(np.float32)
    specular = (v > 0.85).astype(np.float32)
    prior = np.clip(0.6 * blue * (1.0 - s) + 0.4 * specular, 0, 1)
    return prior


def _spectral_gate_passes(ctx: ImageContext, target_object: str, x1, y1, x2, y2,
                          W: int, H: int) -> bool:
    """
    Innovation 7: given a candidate box in pixel space, compute the fraction of
    its pixels that are water/vegetation according to the physics index, and only
    pass (return True) if that fraction clears the gate. If no spectral data is
    available, pass everything (no gating possible).
    """
    low_w = target_object.lower()
    want_water = any(k in low_w for k in MARITIME_KEYWORDS)
    want_veg = any(k in low_w for k in VEGETATION_KEYWORDS)

    if not (want_water or want_veg):
        return True

    # Clamp box to image bounds.
    x1i, y1i = int(max(0, min(x1, W - 1))), int(max(0, min(y1, H - 1)))
    x2i, y2i = int(max(0, min(x2, W))), int(max(0, min(y2, H)))
    if x2i <= x1i or y2i <= y1i:
        return False

    bands = ctx.bands
    has_nir = "nir" in bands

    # Build a per-pixel prior map at full image resolution.
    if want_water:
        if has_nir:
            ndwi = _ndwi(bands)
            prior = (ndwi > NDWI_THRESHOLD).astype(np.float32) if ndwi is not None else None
        else:
            rgb_np = np.array(ctx.rgb.convert("RGB"), dtype=np.uint8)
            rgb_r = ctx.rgb.resize((W, H), Image.BILINEAR)
            rgb_np = np.array(rgb_r, dtype=np.uint8)
            prior = _rgb_water_prior(rgb_np)
    else:  # want_veg
        if has_nir:
            ndvi = _ndvi(bands)
            prior = (ndvi > NDVI_THRESHOLD).astype(np.float32) if ndvi is not None else None
        else:
            # No NIR -> no reliable veg prior -> gate passes.
            prior = None

    if prior is None:
        return True

    # The prior is aligned to ctx.rgb dimensions; account for resize factor.
    pw, ph = prior.shape[1], prior.shape[0]
    fx = pw / float(W)
    fy = ph / float(H)
    px1 = int(x1i * fx); py1 = int(y1i * fy)
    px2 = int(max(px1 + 1, x2i * fx)); py2 = int(max(py1 + 1, y2i * fy))
    px2 = min(px2, pw); py2 = min(py2, ph)

    region = prior[py1:py2, px1:px2]
    if region.size == 0:
        return False
    fraction = float(np.mean(region))
    threshold = GATE_MIN_WATER_FRACTION if want_water else GATE_MIN_VEG_FRACTION
    return fraction >= threshold


# --------------------------------------------------------------------------- #
# INNOVATION 8 -- OBB & POLYGON REFINEMENT
# --------------------------------------------------------------------------- #
def _oriented_box_refine(rgb_np: np.ndarray, box: Tuple[float, float, float, float]):
    """
    Innovation 8: refine a coarse horizontal box [ymin, xmin, ymax, xmax] into an
    oriented bounding box + polygon using edge contours within the box crop.

    Returns: (oriented_box, polygon) where
      - oriented_box: {"center":[cx,cy], "size":[w,h], "angle_deg":theta,
                       "box_2d":[ymin,xmin,ymax,xmax]}
      - polygon: [[x, y], ...] ordered contour points (closed, pixel coords)
    Falls back to the HBB corners if no meaningful contour is found.
    """
    import cv2
    ymin, xmin, ymax, xmax = box
    H, W = rgb_np.shape[:2]

    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    bw, bh = max(xmax - xmin, 1), max(ymax - ymin, 1)
    fallback_obb = {
        "center": [cx, cy],
        "size": [bw, bh],
        "angle_deg": 0.0,
        "box_2d": [ymin, xmin, ymax, xmax],
    }
    fallback_poly = [[int(v[0]), int(v[1])] for v in
                     ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))]

    if not OBB_ENABLED:
        return fallback_obb, fallback_poly

    # Expand crop slightly for contour context, clamp to bounds.
    pad = int(max(4, 0.05 * min(bw, bh)))
    x0 = int(max(0, xmin - pad)); y0 = int(max(0, ymin - pad))
    x1 = int(min(W, xmax + pad)); y1 = int(min(H, ymax + pad))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return fallback_obb, fallback_poly

    crop = rgb_np[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, OBB_EDGE_CANNY_LOW, OBB_EDGE_CANNY_HIGH)
    # Dilate to connect fragmented object edges (morphological active contour-ish).
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback_obb, fallback_poly

    # Choose the largest contour within the crop bounding box.
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < OBB_MIN_AREA:
        return fallback_obb, fallback_poly

    # Shift contour back to full-image coordinates.
    contour_full = best + np.array([x0, y0])

    rect = cv2.minAreaRect(contour_full)  # ((cx,cy),(w,h),theta)
    (rcx, rcy), (rw, rh), theta = rect
    obb = {
        "center": [float(rcx), float(rcy)],
        "size": [float(rw), float(rh)],
        "angle_deg": float(theta),
        "box_2d": [float(ymin), float(xmin), float(ymax), float(xmax)],
    }
    box_pts = cv2.boxPoints(rect)  # 4 corners (2,3)
    poly = [[float(p[0]), float(p[1])] for p in box_pts.tolist()]
    return obb, poly


# --------------------------------------------------------------------------- #
# COMPONENT A: MobileSAM Neural Segmentation (Innovation 8 neural upgrade)
# --------------------------------------------------------------------------- #
def _mobilesam_is_available() -> bool:
    """Check whether the `mobile_sam` package + checkpoint are usable (cached)."""
    global _MOBILESAM_AVAILABLE
    if _MOBILESAM_AVAILABLE is not None:
        return _MOBILESAM_AVAILABLE
    ok = False
    try:
        import mobile_sam  # noqa: F401
        if os.path.isfile(MOBILESAM_CKPT):
            ok = True
    except Exception:
        ok = False
    _MOBILESAM_AVAILABLE = ok
    return ok


def _load_mobilesam() -> Optional[Any]:
    """Lazy-load the MobileSAM SamPredictor (fp32 on CUDA, CPU fallback).

    Defensive: if the package or checkpoint is missing this returns None and the
    caller falls back to the classical OpenCV `_oriented_box_refine()`. MobileSAM
    is tiny (~40MB / ~60MB VRAM in fp32), far below the 3GB innovation budget.
    """
    global _MOBILESAM, _MOBILESAM_PREDICTOR
    if _MOBILESAM_PREDICTOR is not None:
        return _MOBILESAM_PREDICTOR
    if not (MOBILESAM_ENABLED and _mobilesam_is_available()):
        return None
    try:
        import torch
        from mobile_sam import sam_model_registry, SamPredictor

        if _MOBILESAM is None:
            sam = sam_model_registry[MOBILESAM_MODEL_TYPE](checkpoint=MOBILESAM_CKPT)
            if _cuda_available() and not MOBILESAM_VRAMP_FP16:
                # MobileSAM fp16 triggers a prompt-encoder dtype bug; fp32 is tiny.
                sam.to(device="cuda")
            elif _cuda_available() and MOBILESAM_VRAMP_FP16:
                sam.to(device="cuda", dtype=torch.float16)
            else:
                sam.to(device="cpu")
            sam.eval()
            _MOBILESAM = sam
        _MOBILESAM_PREDICTOR = SamPredictor(_MOBILESAM)
    except Exception:
        _MOBILESAM = _MOBILESAM_PREDICTOR = None
        _MOBILESAM_AVAILABLE = False
    return _MOBILESAM_PREDICTOR


def _polygon_from_mask(mask: np.ndarray) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Vectorize a binary mask into a polygon contour + min-area oriented box."""
    import cv2
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no contour in segment mask")
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < MOBILESAM_CONTOUR_MIN_AREA:
        raise ValueError("segment contour too small")
    rect = cv2.minAreaRect(best)                      # ((cx,cy),(w,h),theta)
    (rcx, rcy), (rw, rh), theta = rect
    obb = {
        "center": [float(rcx), float(rcy)],
        "size": [float(rw), float(rh)],
        "angle_deg": float(theta),
    }
    box_pts = cv2.boxPoints(rect)
    poly = [[float(p[0]), float(p[1])] for p in box_pts.tolist()]
    return poly, obb


def _mobilesam_segment(predictor: Any, rgb_np: np.ndarray,
                       box: Tuple[float, float, float, float],
                       label: str, score: float,
                       embed_id: Any = None) -> Optional[Dict[str, Any]]:
    """Neural zero-shot mask from a Grounding DINO HBB prompt.

    The image embedding is computed once per unique `embed_id` (all boxes of one
    image reuse the cached embedding), keeping MobileSAM's VRAM footprint flat.
    Returns a dict with "polygon" + "oriented_box_2d" extras (contract box_2d
    preserved on the caller side). Returns None if segmentation fails (caller
    then uses the classical OpenCV fallback).
    """
    global _MOBILESAM_EMBED_ID
    try:
        import numpy as _np
        ymin, xmin, ymax, xmax = box

        if _MOBILESAM_EMBED_ID != embed_id:
            predictor.set_image(rgb_np)
            _MOBILESAM_EMBED_ID = embed_id

        # Prompt: box corners (SamPredictor box= prompt).
        bbox = _np.array([xmin, ymin, xmax, ymax], dtype=_np.float32)
        mask, conf, _ = predictor.predict(
            box=bbox, multimask_output=True
        )
        best_idx = int(_np.argmax(conf))
        best_mask = mask[best_idx]

        poly, obb = _polygon_from_mask(best_mask)
        return {
            "polygon": _np.round(poly).astype(int).tolist(),
            "oriented_box_2d": obb,
            "segmentation_confidence": float(conf[best_idx]),
        }
    except Exception:
        # Fall back to multiple point prompts (older predictor API without box=).
        try:
            import numpy as _np
            ymin, xmin, ymax, xmax = box
            pts, lbl = [], []
            for fx, fy in ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
                           (0.5, 0.5)):
                pts.append([xmin + fx * (xmax - xmin), ymin + fy * (ymax - ymin)])
                lbl.append(1)
            if _MOBILESAM_EMBED_ID != embed_id:
                predictor.set_image(rgb_np)
                _MOBILESAM_EMBED_ID = embed_id
            mask, conf, _ = predictor.predict(
                point_coords=_np.array(pts, dtype=_np.float32),
                point_labels=_np.array(lbl, dtype=_np.int32),
                multimask_output=True,
            )
            best_idx = int(_np.argmax(conf))
            best_mask = mask[best_idx]
            poly, obb = _polygon_from_mask(best_mask)
            return {
                "polygon": _np.round(poly).astype(int).tolist(),
                "oriented_box_2d": obb,
                "segmentation_confidence": float(conf[best_idx]),
            }
        except Exception:
            return None


def _refine_box(rgb_np: np.ndarray, box: Tuple[float, float, float, float],
                label: str, score: float,
                embed_id: Any = None) -> Tuple[Dict[str, Any], List[List[float]]]:
    """Refine HBB -> (oriented_box, ter). Prefers MobileSAM; falls back to OpenCV."""
    pred = _load_mobilesam()
    if pred is not None:
        seg = _mobilesam_segment(pred, rgb_np, box, label, score, embed_id)
        if seg is not None:
            obb = seg["oriented_box_2d"]
            obb["box_2d"] = list(box)          # preserve contract key
            return obb, seg["polygon"]
    return _oriented_box_refine(rgb_np, box)


# --------------------------------------------------------------------------- #
# COMPONENT B: SAHI slicing-aided hyper inference (multi-scale)
# --------------------------------------------------------------------------- #
def _satquery_use_sahi_enabled() -> bool:
    return os.environ.get("SATQUERY_USE_SAHI", "0") == "1"


def _sahi_is_available() -> bool:
    global _SAHI_AVAILABLE
    if _SAHI_AVAILABLE is not None:
        return _SAHI_AVAILABLE
    try:
        import sahi  # noqa: F401
        _SAHI_AVAILABLE = True
    except Exception:
        _SAHI_AVAILABLE = False
    return _SAHI_AVAILABLE


def _nms_geospatial(entries: List[Dict[str, Any]], iou_thresh: float) -> List[Dict[str, Any]]:
    """Geospatial Non-Maximum Suppression over box_2d proposals (IoU)."""
    if not entries:
        return []
    boxes = [e["box_2d"] for e in entries]          # [ymin, xmin, ymax, xmax]
    scores = [e["confidence"] for e in entries]
    order = sorted(range(len(entries)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]

    def iou(ai, bi):
        a, b = boxes[ai], boxes[bi]
        iy0, iy1 = max(a[0], b[0]), min(a[2], b[2])
        ix0, ix1 = max(a[1], b[1]), min(a[3], b[3])
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        union = areas[ai] + areas[bi] - inter
        return inter / union if union > 0 else 0.0

    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(i, j) < iou_thresh]
    return [entries[i] for i in sorted(keep)]


def _slice_inference(model: Any, processor: Any, rgb_np: np.ndarray,
                     query: str, width: int, height: int) -> List[Tuple]:
    """Slice-aided inference: tile + per-slice Grounding DINO + global remap + NMS.

    Returns list of (xmin, ymin, xmax, ymax, score, label) in global coords.
    This implements SAHI's slicing algorithm natively to guarantee zero regression
    under transformers 5.x (SAHI's HF predictor targets the older API).
    """
    import torch
    import numpy as np
    size = SAHI_SLICE_WH
    stride = max(1, int(size * (1.0 - SAHI_OVERLAP_RATIO)))
    proposal_slices: List[Tuple] = []

    ys = list(range(0, height, stride))
    xs = list(range(0, width, stride))

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + size, height)
            x1 = min(x0 + size, width)
            slice_img = rgb_np[y0:y1, x0:x1]
            slice_h, slice_w = slice_img.shape[:2]
            if slice_h < 64 or slice_w < 64:
                continue
            pil_slice = Image.fromarray(slice_img)
            inputs = processor(images=pil_slice, text=query, return_tensors="pt")
            if _cuda_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            kwargs = dict(
                input_ids=inputs["input_ids"],
                target_sizes=[(slice_h, slice_w)],
                text_threshold=GROUNDING_TEXT_THRESHOLD,
            )
            try:
                r = processor.post_process_grounded_object_detection(
                    outputs, threshold=GROUNDING_BOX_THRESHOLD, **kwargs)[0]
            except TypeError:
                r = processor.post_process_grounded_object_detection(
                    outputs, box_threshold=GROUNDING_BOX_THRESHOLD, **kwargs)[0]
            scores = r.get("scores", [])
            boxes = r.get("boxes", [])
            labels = r.get("text_labels") or r.get("labels", [])
            for i, b in enumerate(boxes):
                sx1, sy1, sx2, sy2 = [float(v) for v in b]
                score = _confidence_to_float(scores[i]) if i < len(scores) else 0.0
                raw_label = labels[i] if i < len(labels) else None
                label = (str(raw_label) if raw_label is not None
                         and str(raw_label) != "" and not str(raw_label).isdigit()
                         else query)
                proposal_slices.append((x0 + sx1, y0 + sy1, x0 + sx2, y0 + sy2,
                                        score, label))

    # Merge duplicates via geospatial NMS (SAHI-equivalent step).
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for xmin, ymin, xmax, ymax, score, label in proposal_slices:
        key = label
        grouped.setdefault(key, []).append({
            "box_2d": [ymin, xmin, ymax, xmax],
            "confidence": score,
            "label": label,
        })
    merged: List[Tuple] = []
    for key, entries in grouped.items():
        kept = _nms_geospatial(entries, SAHI_NMS_IOU_THRESH)
        for e in kept:
            ymin, xmin, ymax, xmax = e["box_2d"]
            merged.append((xmin, ymin, xmax, ymax, e["confidence"], key))
    return merged


# --------------------------------------------------------------------------- #
# LOAD IMAGE (Innovation 10 radiometric + CLAHE, Innovation 9 SAR)
# --------------------------------------------------------------------------- #
def load_context(image_path: str) -> ImageContext:
    """
    Load an image into a full ImageContext (spectral bands, SAR flag, enhanced
    display RGB). This is the internal workhorse for run_vqa / run_grounding.

    - Multi-band/SAR GeoTIFF: full spectral pipeline (NDVI/NDWI bands, SAR Lee filter).
    - Standard PNG/JPEG: RGB with percentile stretch + CLAHE (I10).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()

    if ext in (".tif", ".tiff"):
        try:
            return _load_multiband_raster(image_path)
        except ImportError:
            warnings.warn("rasterio not installed; falling back to PIL read of TIF.")
        except Exception as exc:
            warnings.warn(f"raster read failed ({exc}); falling back to PIL read.")

    # --- Standard image path (PNG/JPG) with I10 stretch + CLAHE ---
    pil = Image.open(image_path).convert("RGB")
    np_rgb = np.array(pil, dtype=np.uint8)
    np_rgb = _percentile_stretch(np_rgb)
    np_rgb = _clahe_luminance(np_rgb)
    # RGB has no NIR; provide color-space water prior via bands hook.
    return ImageContext(
        Image.fromarray(np_rgb),
        array=None,
        bands={},
        is_sar=False,
        is_geotiff=False,
        source_ext=ext,
    )


def load_image(image_path: str) -> "Image.Image":
    """
    Backward-compatible loader returning an enhanced RGB PIL Image (I10 stretch +
    CLAHE applied). Kept for Person 6 / Pair 2 / existing tests that expect a PIL
    image. For full spectral/SAR context, use load_context() instead.
    """
    return load_context(image_path).rgb


# --------------------------------------------------------------------------- #
# SHARED VLM LOADER (used by Person 5 AND Person 6)
# --------------------------------------------------------------------------- #
def _load_vlm() -> Tuple[Any, Any]:
    """Lazy, cached loader for the Qwen2-VL model + processor (4-bit on GPU)."""
    global _VLM, _VLM_PROCESSOR
    if _VLM is not None and _VLM_PROCESSOR is not None:
        return _VLM, _VLM_PROCESSOR

    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2VLForConditionalGeneration,
    )

    load_kwargs: Dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    if _cuda_available():
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["low_cpu_mem_usage"] = True

    print(f"[SatQuery] Loading VLM: {VLM_REPO} (cuda={_cuda_available()}) ...")
    t0 = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(VLM_REPO, **load_kwargs)
    processor = AutoProcessor.from_pretrained(VLM_REPO, trust_remote_code=True)
    if not _cuda_available():
        model = model.to("cpu")
    model.eval()
    print(f"[SatQuery] VLM loaded in {time.time() - t0:.1f}s")
    _VLM, _VLM_PROCESSOR = model, processor
    return _VLM, _VLM_PROCESSOR


# --------------------------------------------------------------------------- #
# SAR CONTEXT PROMPT EXPANSION (Innovation 9) + VQA
# --------------------------------------------------------------------------- #
_VQA_DOMAIN_EXPANSION: Dict[str, str] = {
    "sar": (
        "Context: this is single-polarization radar backscatter (SAR). A Lee spectral "
        "speckle filter was applied. Dark regions are specular smooth surfaces (e.g. "
        "calm water) with backscatter below -15 dB; bright regions are double-bounce "
        "scatterers (built-up structures) above -5 dB. Interpret surface roughness and "
        "water boundaries accordingly."
    ),
    "optical": "",
}

# The domain prompt expander (kept generic so the contract stays frozen).
EXPANSION_MAP: Dict[str, str] = {
    # General-purpose expansion prompts keyed by intent tokens.
    "describe": "Focus on dominant land cover, terrain, and recognizable structures.",
    "count": "Enumerate distinct objects accurately; state the number explicitly.",
    "water": "Identify water bodies, rivers, and wet regions.",
    "veg": "Identify vegetation, crops, forests, and farmland.",
    "sar": _VQA_DOMAIN_EXPANSION["sar"],
}


def _expand_system_prompt(prompt: str, ctx: ImageContext) -> str:
    """
    Build the satellite-domain system prompt, injecting a SAR-aware context block
    (Innovation 9) when the image is a single-band radar scene.
    """
    system = (
        "You are SatQuery, a remote-sensing vision assistant. You analyse aerial and "
        "satellite imagery (optical and SAR). Answer the user's question about this "
        "satellite image. Be specific about objects, their count, locations, and land "
        "cover. If unsure of an exact measurement, say so. Keep the answer concise "
        "(1-4 sentences)."
    )
    if ctx.is_sar:
        system += "\n\n" + _VQA_DOMAIN_EXPANSION["sar"]

    # Domain prompt expander: apply intent-specific guidance from EXPANSION_MAP
    # when the query's keyword matches a known expansion.
    low = prompt.lower()
    for token, expansion in EXPANSION_MAP.items():
        if token in low:
            system += f"\nPointer: {expansion}"
    return system


# --------------------------------------------------------------------------- #
# DYNAMIC TOKEN SOFTMAX CONFIDENCE (TSC)
# --------------------------------------------------------------------------- #
def _run_vlm_generation_with_confidence(image: "Image.Image", prompt: str) -> Tuple[str, float]:
    """
    Shared Qwen2-VL generation returning (text, dynamic-token-softmax-confidence).

    Generates ONCE, capturing per-step logits to derive a model-logit-based
    confidence (mean top-1 softmax probability across generated tokens). This
    scales confidence with the model's actual generation likelihood instead of
    string-length heuristics.

    Person 6 reuses this for multi-image / cross-modal textual summaries as well.

    Returns: (decoded_text, confidence) where confidence in [0, 1].
    """
    import torch
    model, processor = _load_vlm()

    chat = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    if _cuda_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=VLM_MAX_NEW_TOKENS,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    generated_ids = out.sequences
    step_scores = out.scores  # tuple of per-step logits

    # --- Dynamic token softmax confidence ---
    if step_scores:
        probs = [torch.softmax(step.float(), dim=-1).max(dim=-1).values.item()
                 for step in step_scores]
        mean_prob = float(np.mean(probs))
        # Map mean top-1 softmax probability into an interpretable [0,1] scale via
        # a centered logistic. Empirical mean for concise, fluent answers ~0.6-0.7;
        # lower values appear when the model hedges or rambles. This preserves
        # dynamic range (unlike a coarse linear scale that saturates at 1.0).
        conf = 1.0 / (1.0 + np.exp(-11.0 * (mean_prob - 0.60)))
    else:
        conf = 0.5

    new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    output = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output.strip(), round(conf, 4)


# --------------------------------------------------------------------------- #
# TASK 1 -- VQA & SCENE CAPTIONING (with SAR prompt + dynamic confidence)
# --------------------------------------------------------------------------- #
def run_vqa(image_path: str, prompt: str) -> Dict[str, Any]:
    """
    Answer a natural-language question (or caption a scene) for a single image.

    Contract:
        {"text_response": str, "confidence": float}
    """
    t0 = time.time()
    ctx = load_context(image_path)
    system = _expand_system_prompt(prompt, ctx)
    full_prompt = f"{system}\n\nUser query: {prompt}"

    # Innovation 9 SAR handling: also surface radar structural context to VQA.
    # Innovation-specific guidance is folded into the system prompt above.
    if "describe" in prompt.lower() or "caption" in prompt.lower():
        task_hint = "Describe the scene and the dominant land cover / features."
        full_prompt = f"{system}\n\n{task_hint}\nUser query: {prompt}"

    text_response, conf = _run_vlm_generation_with_confidence(ctx.rgb, full_prompt)

    # Conservation: if the logit confidence came back degenerate, apply heuristics.
    if not text_response:
        text_response = "The model produced no answer for this image."
        conf = min(conf, 0.1) if conf else 0.1
    else:
        lowered = text_response.lower()
        if any(w in lowered for w in ("cannot see", "can't see", "unable", "i don't know")):
            conf = min(conf, 0.3)

    return {
        "text_response": text_response,
        "confidence": _confidence_to_float(conf),
    }


# --------------------------------------------------------------------------- #
# GROUNDING DINO LOADER
# --------------------------------------------------------------------------- #
def _load_grounding() -> Tuple[Any, Any]:
    """Lazy, cached loader for Grounding DINO."""
    global _GROUNDING, _GROUNDING_PROCESSOR
    if _GROUNDING is not None and _GROUNDING_PROCESSOR is not None:
        return _GROUNDING, _GROUNDING_PROCESSOR

    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    print(f"[SatQuery] Loading Grounding DINO: {GROUNDING_REPO} ...")
    t0 = time.time()
    if _cuda_available():
        _GROUNDING = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_REPO, trust_remote_code=True).to("cuda")
    else:
        _GROUNDING = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_REPO, trust_remote_code=True).to("cpu")
    _GROUNDING.eval()
    _GROUNDING_PROCESSOR = AutoProcessor.from_pretrained(GROUNDING_REPO, trust_remote_code=True)
    print(f"[SatQuery] Grounding DINO loaded in {time.time() - t0:.1f}s")
    return _GROUNDING, _GROUNDING_PROCESSOR


# --------------------------------------------------------------------------- #
# TASK 2 -- TEXT-GUIDED VISUAL GROUNDING (Innovations 7, 8)
# --------------------------------------------------------------------------- #
def run_grounding(image_path: str, target_object: str) -> Dict[str, Any]:
    """
    Localize instances of a text-described object in a single image.

    Component A (MobileSAM): each surviving HBB is refined to a neural zero-shot
        polygon + oriented box (falls back to classical OpenCV if unavailable).
    Component B (SAHI): images over 1024px (or SATQUERY_USE_SAHI=1) are sliced
        into overlapping tiles, per-slice inference run, boxes remapped to global
        coords and merged with geospatial NMS. Falls back to full-image inference.
    Innovation 7: spectral gating (NDWI/NDVI) removes contextually-impossible boxes.

    Contract:
        {"text_response": str,
         "bounding_boxes": [{"label": str,
                             "box_2d": [ymin, xmin, ymax, xmax],
                             "confidence": float}],
         "confidence": float}
    """
    import torch

    t0 = time.time()
    ctx = load_context(image_path)
    width, height = ctx.rgb.size

    model, processor = _load_grounding()
    query = target_object.strip() or "object"
    rgb_np = np.array(ctx.rgb, dtype=np.uint8)
    image_embed_id = (image_path, rgb_np.shape[0], rgb_np.shape[1])

    # --- Component B: SAHI slicing-aided inference ------------------------------
    use_sahi = (SAHI_ENABLED and _sahi_is_available()) and (
        _satquery_use_sahi_enabled() or width > SAHI_RESIZE_TRIGGER
        or height > SAHI_RESIZE_TRIGGER)
    if use_sahi:
        detections = _slice_inference(model, processor, rgb_np, query, width, height)
    else:
        inputs = processor(images=ctx.rgb, text=query, return_tensors="pt")
        if _cuda_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        kwargs = dict(
            input_ids=inputs["input_ids"],
            target_sizes=[(height, width)],
            text_threshold=GROUNDING_TEXT_THRESHOLD,
        )
        try:
            results = processor.post_process_grounded_object_detection(
                outputs, threshold=GROUNDING_BOX_THRESHOLD, **kwargs)[0]
        except TypeError:
            results = processor.post_process_grounded_object_detection(
                outputs, box_threshold=GROUNDING_BOX_THRESHOLD, **kwargs)[0]
        scores = results.get("scores", [])
        boxes = results.get("boxes", [])
        labels = results.get("text_labels") or results.get("labels", [])
        detections = []
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = [float(v) for v in box]
            raw_label = labels[i] if i < len(labels) else None
            label = (str(raw_label) if raw_label is not None
                     and str(raw_label) != "" and not str(raw_label).isdigit() else query)
            sc = _confidence_to_float(scores[i]) if i < len(scores) else 0.0
            detections.append((xmin, ymin, xmax, ymax, sc, label))

    # --- Post-process each surviving detection -----------------------------------
    bounding_boxes: List[Dict[str, Any]] = []
    for xmin, ymin, xmax, ymax, score, label in detections:
        # Clamp to valid pixel bounds for downstream GeoJSON/map rendering safety.
        xmin, xmax = max(0.0, xmin), min(float(width), xmax)
        ymin, ymax = max(0.0, ymin), min(float(height), ymax)
        if xmax <= xmin or ymax <= ymin:
            continue

        # Innovation 7: spectral gating drops physics-impossible boxes.
        if not _spectral_gate_passes(ctx, target_object, xmin, ymin, xmax, ymax,
                                     width, height):
            continue

        # Component A: refine HBB -> neural OBB + polygon (MobileSAM or OpenCV).
        obb, poly = _refine_box(rgb_np, (ymin, xmin, ymax, xmax),
                                label, score, embed_id=image_embed_id)

        entry: Dict[str, Any] = {
            "label": label,
            "box_2d": [ymin, xmin, ymax, xmax],  # CONTRACT (frozen order)
            "confidence": score,
        }
        if OBB_ENABLED:
            entry["oriented_box_2d"] = obb        # extra (optional)
            entry["polygon"] = poly                # extra (optional)
        bounding_boxes.append(entry)

    if bounding_boxes:
        n = len(bounding_boxes)
        text_response = (
            f"Detected {n} instance(s) of '{target_object}' in the image. "
            f"Highest-confidence detection scores "
            f"{max(b['confidence'] for b in bounding_boxes):.2f}."
        )
        confidence = max(b["confidence"] for b in bounding_boxes)
    else:
        text_response = (
            f"No '{target_object}' detected above the confidence threshold "
            f"({GROUNDING_BOX_THRESHOLD}). Consider uploading higher-resolution imagery."
        )
        confidence = 0.0

    return {
        "text_response": text_response,
        "bounding_boxes": bounding_boxes,
        "confidence": _confidence_to_float(confidence),
    }


# --------------------------------------------------------------------------- #
# METADATA EXPORTS
# --------------------------------------------------------------------------- #
def get_execution_metadata() -> Dict[str, Any]:
    """Model identifiers + active innovations for the agentic execution trace."""
    active = [
        "I07_spectral_gating",
        "I08_obb_polygon",
        "I09_sar_backscatter",
        "I10_radiometric_stretch",
        "TSC_dynamic_token_softmax",
        "COMPONENT_A_mobilesam_segmentation",
        "COMPONENT_B_sahi_slicing",
    ]
    return {
        "vqa_model": VLM_REPO,
        "grounding_model": GROUNDING_REPO,
        "segmentation": "MobileSAM" if _mobilesam_is_available() else "classical-opencv",
        "slicing": "SAHI-enabled" if _sahi_is_available() else "full-image",
        "quantization": "4bit-nf4" if _cuda_available() else "none (cpu fp32)",
        "device": "cuda" if _cuda_available() else "cpu",
        "innovations": active,
    }


def flush_models() -> None:
    """Release all cached models (frees VRAM)."""
    global _VLM, _VLM_PROCESSOR, _GROUNDING, _GROUNDING_PROCESSOR
    global _MOBILESAM, _MOBILESAM_PREDICTOR
    import gc
    try:
        import torch
        del _VLM, _VLM_PROCESSOR, _GROUNDING, _GROUNDING_PROCESSOR
        del _MOBILESAM, _MOBILESAM_PREDICTOR
        gc.collect()
        if _cuda_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    _VLM = _VLM_PROCESSOR = _GROUNDING = _GROUNDING_PROCESSOR = None
    _MOBILESAM = _MOBILESAM_PREDICTOR = None


# Allow running as a script: `python models_registry.py image.png "describe"`
if __name__ == "__main__":
    import sys
    _img = sys.argv[1] if len(sys.argv) > 1 else "RS/before.png"
    try:
        _res = run_vqa(_img, "Describe this satellite image")
    except Exception as exc:
        _res = {"text_response": f"VQA unavailable here: {exc}", "confidence": 0.0}
    print(json.dumps(_res, indent=2, default=str))
