"""Bi-temporal change analysis specialist (two DATES, same modality).

Implements PLAN.md §7 W4: real change detection on a code path distinct from
fusion.py (R3). Fusion ingests two modalities of one date; this ingests two
dates of one modality. The paths never call each other (§5.3) and share no
code beyond satquery.io/satquery.adapters (W0/W2-owned).

SCOPE (PLAN.md §8 fallback line, exercised deliberately — see docs/status/W4.md):
CDVQA and SECOND (PLAN.md §7's named acceptance data) are Google Drive/Baidu
hosted with no direct HTTP link and are NOT fetched here (W1-owned, and W1
could not obtain them — docs/status/W1.md). This module is classical CV
(registration + radiometric normalisation + differencing) plus `benclip`
labels read on BOTH dates, with NO trained change head — the documented
fallback. R3 is about the capability (a real two-date code path), not a
CDVQA benchmark score.

W4 implementation notes (PLAN.md §2.5, §4.1, §4.5):
  - Registration: skip entirely when both inputs are already co-registered
    (identical CRS + affine transform via satquery.io.raster) — see
    `_already_coregistered`. Otherwise fall back to ORB + affine, with every
    §2.5 bug fixed: the affine is fit on the FILTERED (`good`) match list,
    not the raw one; `cv2.estimateAffinePartial2D` is null-checked; empty/None
    ORB descriptors are guarded. A featureless scene (uniform water/farmland)
    degrades to a low-confidence no-warp fallback rather than crashing.
  - Radiometric normalisation (skimage histogram matching) runs before
    differencing, so illumination drift is not read as change.
  - The mask is differencing + morphological open/close, saved to disk.
  - `benclip.predict_labels()` runs on BOTH raw dates (never the warped
    array — that's for the pixel mask only); band logic is entirely
    benclip's (§4.5), never re-implemented here. The two label sets are
    turned into a softmax-normalised per-class CONFIDENCE delta — a shift in
    benclip's read of the scene, not a measured land-cover area change; the
    evidence keys say so explicitly.
  - Every returned number is coerced with float() (§4.5) before validation.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from satquery.contracts import ChangeResult, validate_change_result
from satquery.io.raster import RasterInput, load_raster

# ORB match-distance cutoff below which a match is trusted ("good"). Mirrors
# old_files/BT_CM.py's empirically-chosen threshold for Hamming distance on
# 256-bit ORB descriptors; kept because it is a reasonable, well-known cutoff
# for this descriptor, not because the old file was trusted verbatim (its
# bugs below are all fixed).
_ORB_GOOD_DISTANCE = 50.0
# Minimum good matches needed to fit an affine (2D affine has 4-6 DOF).
_MIN_GOOD_MATCHES = 3
# registration_confidence saturates at 1.0 once this many good matches exist.
_CONFIDENCE_SATURATION_MATCHES = 25.0


def _run_output_dir(t0_path: str, t1_path: str) -> str:
    """A per-pair, deterministic runs/ subdir so repeated calls on different
    pairs don't clobber each other's masks/overlays."""
    key = hashlib.sha1(
        f"{os.path.abspath(t0_path)}|{os.path.abspath(t1_path)}".encode()
    ).hexdigest()[:12]
    return os.path.join("runs", "change", key)


def _to_gray_u8(raster: RasterInput) -> np.ndarray:
    """Grayscale uint8 view of a RasterInput's always-safe display_rgb."""
    return cv2.cvtColor(raster.display_rgb, cv2.COLOR_RGB2GRAY)


def _already_coregistered(r0: RasterInput, r1: RasterInput) -> bool:
    """PLAN.md W4 task 1: skip registration when both inputs are
    georeferenced and already share the same CRS, transform and shape."""
    if not (r0.is_georeferenced and r1.is_georeferenced):
        return False
    if r0.crs is None or r1.crs is None or r0.crs != r1.crs:
        return False
    if r0.transform is None or r1.transform is None:
        return False
    if r0.transform != r1.transform:
        return False
    return r0.array.shape[:2] == r1.array.shape[:2]


def _orb_register(g0: np.ndarray, g1: np.ndarray) -> Dict[str, Any]:
    """Feature-based registration of t1 onto t0's pixel grid.

    Fixes every PLAN.md §2.5 bug found in old_files/BT_CM.register_images:
      - the affine is fit on the FILTERED `good` matches, not the raw list
      - cv2.estimateAffinePartial2D's return is null-checked
      - empty/None ORB descriptors are guarded before they reach BFMatcher

    Returns a dict with keys: mechanism, aligned (np.ndarray, g0's shape),
    confidence, good_matches, keypoints_t0, keypoints_t1, reason.
    """
    h0, w0 = g0.shape[:2]

    def _fallback(confidence: float, good_matches: int, n_kp0: int, n_kp1: int, reason: str) -> Dict[str, Any]:
        aligned = g1 if g1.shape[:2] == (h0, w0) else cv2.resize(g1, (w0, h0))
        return {
            "mechanism": "no_warp_fallback",
            "aligned": aligned,
            "confidence": float(confidence),
            "good_matches": int(good_matches),
            "keypoints_t0": int(n_kp0),
            "keypoints_t1": int(n_kp1),
            "reason": reason,
        }

    orb = cv2.ORB_create(nfeatures=2500)
    kp0, des0 = orb.detectAndCompute(g0, None)
    kp1, des1 = orb.detectAndCompute(g1, None)
    n_kp0 = len(kp0) if kp0 is not None else 0
    n_kp1 = len(kp1) if kp1 is not None else 0

    # §2.5 bug 3: guard empty/None ORB descriptors before BFMatcher, which
    # crashes on either. This is exactly where a featureless scene (uniform
    # water/farmland) dies in old_files/BT_CM.py.
    if des0 is None or des1 is None or len(des0) == 0 or len(des1) == 0:
        return _fallback(
            0.05, 0, n_kp0, n_kp1,
            "no ORB descriptors on one or both frames (featureless/low-texture scene)",
        )

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(des1, des0)
    except cv2.error as exc:
        return _fallback(0.05, 0, n_kp0, n_kp1, f"ORB matching failed: {exc}")

    matches = sorted(matches, key=lambda m: m.distance)
    good = [m for m in matches if m.distance < _ORB_GOOD_DISTANCE]
    confidence = min(len(good) / _CONFIDENCE_SATURATION_MATCHES, 1.0)

    if len(good) < _MIN_GOOD_MATCHES:
        return _fallback(
            min(confidence, 0.15), len(good), n_kp0, n_kp1,
            f"only {len(good)} good matches (< {_MIN_GOOD_MATCHES} needed to fit an affine)",
        )

    # §2.5 bug 1: fit the affine on the FILTERED `good` list, not `matches`.
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp0[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, _inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    # §2.5 bug 2: null-check before warpAffine, which crashes on None.
    if matrix is None:
        return _fallback(
            min(confidence, 0.15), len(good), n_kp0, n_kp1,
            "cv2.estimateAffinePartial2D returned None (degenerate point configuration)",
        )

    aligned = cv2.warpAffine(g1, matrix, (w0, h0))
    return {
        "mechanism": "orb_affine",
        "aligned": aligned,
        "confidence": float(confidence),
        "good_matches": int(len(good)),
        "keypoints_t0": int(n_kp0),
        "keypoints_t1": int(n_kp1),
        "reason": f"{len(good)} good ORB matches, affine fit succeeded",
    }


def _normalize_and_diff(g0: np.ndarray, aligned_g1: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """Radiometric normalisation (histogram matching) + differencing +
    morphological cleanup. PLAN.md W4 tasks 3-4."""
    from skimage.exposure import match_histograms

    matched = match_histograms(aligned_g1.astype(np.float64), g0.astype(np.float64))
    matched_u8 = np.clip(matched, 0, 255).astype(np.uint8)

    diff = cv2.absdiff(g0, matched_u8)
    _thresh, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    changed_area_fraction = float(np.mean(mask > 0))
    diff_stats = {
        "diff_mean": float(np.mean(diff)),
        "diff_max": float(np.max(diff)),
        "otsu_threshold": float(_thresh),
    }
    return mask, changed_area_fraction, diff_stats


def _save_mask(mask: np.ndarray, output_dir: str) -> Optional[str]:
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "change_mask.png")
        Image.fromarray(mask.astype(np.uint8)).save(path)
        return path
    except Exception:
        return None


def _render_overlay(t0_raster: RasterInput, mask: np.ndarray, output_dir: str) -> Optional[str]:
    """Red change-mask overlay on the t0 display image. PLAN.md W4 task 6."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        base = t0_raster.display_rgb
        h, w = base.shape[:2]
        mask_r = mask if mask.shape[:2] == (h, w) else cv2.resize(
            mask, (w, h), interpolation=cv2.INTER_NEAREST
        )
        alpha = (mask_r > 0).astype(np.float32)[..., None] * 0.5
        red = np.zeros_like(base, dtype=np.float32)
        red[..., 0] = 255.0
        blended = base.astype(np.float32) * (1.0 - alpha) + red * alpha
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        path = os.path.join(output_dir, "change_overlay.png")
        Image.fromarray(blended).save(path)
        return path
    except Exception:
        return None


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    total = np.sum(e)
    if total <= 0 or not np.isfinite(total):
        return np.full_like(x, 1.0 / max(len(x), 1))
    return e / total


def _per_class_delta(
    labels_t0: List[Dict[str, Any]], labels_t1: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Structured per-class benclip delta between the two dates.

    IMPORTANT (honesty, PLAN.md §5.9): the underlying `score` from
    `benclip.predict_labels` is a raw image-text cosine similarity, not a
    calibrated probability or a physical land-cover area fraction. This
    function softmaxes each date's scores into a confidence distribution
    over classes (summing to 100%) so the delta is expressed in comparable
    percentage points across the two dates — it is a shift in benclip's
    read of the scene, never claimed as a measured area change.
    """
    scores0 = {l["label"]: float(l["score"]) for l in labels_t0}
    scores1 = {l["label"]: float(l["score"]) for l in labels_t1}
    all_labels = sorted(set(scores0) | set(scores1))
    if not all_labels:
        return []

    raw0 = np.array([scores0.get(name, 0.0) for name in all_labels], dtype=np.float64)
    raw1 = np.array([scores1.get(name, 0.0) for name in all_labels], dtype=np.float64)
    pct0 = _softmax(raw0) * 100.0
    pct1 = _softmax(raw1) * 100.0

    deltas = [
        {
            "label": name,
            "raw_score_t0": float(raw0[i]),
            "raw_score_t1": float(raw1[i]),
            "confidence_pct_t0": float(pct0[i]),
            "confidence_pct_t1": float(pct1[i]),
            "delta_confidence_pct": float(pct1[i] - pct0[i]),
        }
        for i, name in enumerate(all_labels)
    ]
    deltas.sort(key=lambda d: abs(d["delta_confidence_pct"]), reverse=True)
    return deltas


def _predict_both_dates(
    r0: RasterInput, r1: RasterInput, evidence: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Run benclip.predict_labels on BOTH raw dates and write diagnostics
    into `evidence`. Returns the per-class delta (empty if either date's
    prediction failed).

    Uses benclip's own module-level singleton (`bc=None`) rather than
    reloading a fresh checkpoint per call (measured ~7.5s/load) — the
    singleton lives inside satquery.adapters.benclip (§4.3 exempts benclip
    from the modelpool, and the cache is the adapter's own, not a
    specialist-owned module global). This is a deliberate deviation from
    fusion.py's fresh-load-per-call idiom; see docs/status/W4.md.
    `top_k=64` is safely larger than benclip's 19-class vocabulary, so both
    calls return every class's score (`predict_labels` just slices
    `argsort(...)[:top_k]`) — needed so the delta below is symmetric.
    """
    try:
        from satquery.adapters.benclip import predict_labels
    except ImportError as exc:
        evidence["benclip_unavailable"] = str(exc)
        return []

    labels0 = labels1 = None
    try:
        res0 = predict_labels(r0, top_k=64)
        labels0 = res0["labels"]
        evidence["benclip_t0_labels"] = [
            {"label": l["label"], "score": float(l["score"])} for l in labels0[:5]
        ]
        evidence["band_mapping_t0"] = res0["band_mapping"]
    except (FileNotFoundError, RuntimeError) as exc:
        evidence["benclip_t0_error"] = str(exc)

    try:
        res1 = predict_labels(r1, top_k=64)
        labels1 = res1["labels"]
        evidence["benclip_t1_labels"] = [
            {"label": l["label"], "score": float(l["score"])} for l in labels1[:5]
        ]
        evidence["band_mapping_t1"] = res1["band_mapping"]
    except (FileNotFoundError, RuntimeError) as exc:
        evidence["benclip_t1_error"] = str(exc)

    if labels0 is None or labels1 is None:
        return []
    return _per_class_delta(labels0, labels1)


def _build_text_response(
    r0: RasterInput,
    r1: RasterInput,
    registration: Dict[str, Any],
    changed_area_fraction: float,
    per_class_delta: List[Dict[str, Any]],
    benclip_error: Optional[str],
) -> str:
    mech = registration["mechanism"]
    if mech == "skip_already_coregistered":
        reg_desc = "registration skipped (t0/t1 already share CRS and transform)"
    elif mech == "orb_affine":
        reg_desc = (
            f"ORB feature-based affine registration ({registration['good_matches']} good "
            f"matches, registration confidence {registration['confidence']:.2f})"
        )
    else:
        reg_desc = (
            f"registration fallback, no warp applied ({registration['reason']}; "
            f"registration confidence {registration['confidence']:.2f})"
        )

    parts = [
        f"Bi-temporal change analysis, t0 {r0.modality} ({r0.band_count}-band) vs "
        f"t1 {r1.modality} ({r1.band_count}-band). {reg_desc}.",
        f"Differencing + morphological cleanup flags "
        f"{changed_area_fraction * 100.0:.1f}% of pixels as changed.",
    ]
    if per_class_delta:
        movers = ", ".join(
            f"{d['label']} {d['delta_confidence_pct']:+.1f}pp"
            for d in per_class_delta[:3]
        )
        parts.append(
            f"Largest benclip class-confidence shifts between the two dates: {movers}."
        )
    elif benclip_error:
        parts.append(
            f"Note: benclip unavailable ({benclip_error}); change evidence relies on "
            f"pixel differencing only."
        )
    return " ".join(parts)


def _compute_confidence(registration_confidence: float, mask_written: bool, has_delta: bool) -> float:
    parts = [
        registration_confidence,
        0.8 if has_delta else 0.2,
        1.0 if mask_written else 0.0,
    ]
    return float(np.clip(np.mean(parts), 0.0, 1.0))


def _error_result(message: str, t0_path: str, t1_path: str) -> ChangeResult:
    return validate_change_result({
        "text_response": f"Change analysis failed: {message}",
        "change_mask_path": None,
        "overlay_path": None,
        "metrics": {},
        "confidence": 0.0,
        "confidence_basis": "heuristic",
        "evidence": {"error": message, "t0_file": t0_path, "t1_file": t1_path},
    })


def run_change(image_path_t0: str, image_path_t1: str, query: str) -> ChangeResult:
    """Analyse change between two dates of the same modality. PLAN.md §4.1.

    Pipeline: load both rasters -> skip-or-ORB registration -> radiometric
    normalisation -> differencing + morphological cleanup -> mask/overlay to
    disk -> benclip.predict_labels on BOTH raw dates -> per-class confidence
    delta. Never imports another specialist (§5.3); all band logic lives in
    benclip (§4.5).
    """
    try:
        r0 = load_raster(image_path_t0)
    except Exception as exc:
        return _error_result(f"failed to load t0 raster {image_path_t0!r}: {exc}", image_path_t0, image_path_t1)
    try:
        r1 = load_raster(image_path_t1)
    except Exception as exc:
        return _error_result(f"failed to load t1 raster {image_path_t1!r}: {exc}", image_path_t0, image_path_t1)

    output_dir = _run_output_dir(image_path_t0, image_path_t1)

    evidence: Dict[str, Any] = {
        "t0_file": image_path_t0,
        "t0_modality": r0.modality,
        "t0_bands": int(r0.band_count),
        "t1_file": image_path_t1,
        "t1_modality": r1.modality,
        "t1_bands": int(r1.band_count),
    }

    g0 = _to_gray_u8(r0)
    g1 = _to_gray_u8(r1)

    if _already_coregistered(r0, r1):
        registration: Dict[str, Any] = {
            "mechanism": "skip_already_coregistered",
            "confidence": 1.0,
            "good_matches": None,
            "keypoints_t0": None,
            "keypoints_t1": None,
            "reason": "identical CRS and affine transform reported by load_raster",
        }
        aligned_g1 = g1
    else:
        reg = _orb_register(g0, g1)
        aligned_g1 = reg.pop("aligned")
        registration = reg

    # Safety net: harmonise shape if registration fell back without warping
    # onto a differently-sized frame (two arbitrary real scenes need not
    # share pixel dimensions).
    if aligned_g1.shape[:2] != g0.shape[:2]:
        aligned_g1 = cv2.resize(aligned_g1, (g0.shape[1], g0.shape[0]))

    evidence["registration"] = registration
    registration_confidence = float(registration["confidence"])

    try:
        mask, changed_area_fraction, diff_stats = _normalize_and_diff(g0, aligned_g1)
        evidence["diff_stats"] = diff_stats
    except Exception as exc:
        mask = np.zeros_like(g0)
        changed_area_fraction = 0.0
        evidence["mask_error"] = str(exc)

    mask_path = _save_mask(mask, output_dir)
    overlay_path = _render_overlay(r0, mask, output_dir)
    evidence["mask_stats"] = {
        "changed_pixels": int(np.sum(mask > 0)),
        "total_pixels": int(mask.size),
    }

    per_class_delta = _predict_both_dates(r0, r1, evidence)
    evidence["per_class_delta"] = per_class_delta

    benclip_note = (
        evidence.get("benclip_unavailable")
        or evidence.get("benclip_t0_error")
        or evidence.get("benclip_t1_error")
    )

    metrics: Dict[str, Any] = {
        "registration_confidence": registration_confidence,
        "changed_area_fraction": float(changed_area_fraction),
        "per_class_delta": per_class_delta,
    }

    text_response = _build_text_response(
        r0, r1, registration, changed_area_fraction, per_class_delta, benclip_note,
    )
    confidence = _compute_confidence(registration_confidence, mask_path is not None, bool(per_class_delta))

    result: ChangeResult = {
        "text_response": text_response,
        "change_mask_path": mask_path,
        "overlay_path": overlay_path,
        "metrics": metrics,
        "confidence": float(confidence),
        "confidence_basis": "heuristic",
        "evidence": evidence,
    }
    return validate_change_result(result)
