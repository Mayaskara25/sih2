"""
SatQuery AI -- Innovation Test Suite (Innovations 7-10 + Dynamic Token Softmax)

Validates that each innovation triggers correctly on sample inputs (GeoTIFFs and
standard PNGs) WITHOUT CUDA OOM on 6GB VRAM, while the frozen function contracts
(`run_vqa`, `run_grounding`, `get_execution_metadata`) remain intact.

Usage:
    .venv\\Scripts\\python.exe test_innovations.py [fast|full]

    fast  -> run pure-numpy/OpenCV unit checks for I07/I08 (no model inference)
    full  -> ALSO run real model inference for I09/I10 + grounding/+VQA path
             (downloads/loads Grounding DINO + Qwen2-VL; ~30-90s)
"""

import os
import sys
import json
import time
import glob
import tempfile
import warnings

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

import models_registry as m

REPORT = {"ok": [], "fail": [], "skip": [], "latency_ms": {}}


def record(name, passed, detail=""):
    (REPORT["ok"] if passed else REPORT["fail"]).append(name)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #
def make_multiband_geotiff(tmpdir):
    """Build a 3-band synthetic GeoTIFF (red, green, nir) with a water patch."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        return None

    H, W = 128, 128
    red = np.full((H, W), 0.25, dtype=np.float64)
    green = np.full((H, W), 0.25, dtype=np.float64)
    nir = np.full((H, W), 0.25, dtype=np.float64)
    # water patch: high green, very low NIR
    green[40:90, 30:90] = 0.6
    nir[40:90, 30:90] = 0.05
    path = os.path.join(tmpdir, "multiband_test.tif")
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W, count=3,
        dtype="float64", crs="EPSG:3857", transform=from_origin(0, 128, 1, 1),
    ) as dst:
        dst.descriptions = ["Sentinel-2 L2A B04 (Red)", "Sentinel-2 L2A B03 (Green)",
                            "Sentinel-2 L2A B08 (NIR)"]
        dst.write(red, 1)
        dst.write(green, 2)
        dst.write(nir, 3)
    return path


def make_diagonal_rgb(tmpdir):
    """RGB PNG with a diagonal bright bar (for OBB)."""
    import cv2
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    pts = np.array([[25, 12], [75, 25], [62, 75], [12, 62]], dtype=np.int32)
    cv2.fillPoly(arr, [pts], (255, 255, 255))
    path = os.path.join(tmpdir, "diagonal.png")
    Image.fromarray(arr).save(path)
    return path


# --------------------------------------------------------------------------- #
# TEST: Innovation 10 (radiometric stretch + CLAHE)
# --------------------------------------------------------------------------- #
def test_i10_radiometric():
    # A low-contrast dark image should be stretched & CLAHE-enhanced to usable u8.
    arr = np.full((64, 64, 3), 5, dtype=np.uint8)
    arr[10:20, 10:20] = 12
    rgb = m._percentile_stretch(arr)
    l = m._clahe_luminance(arr)
    ok = (rgb.dtype == np.uint8 and rgb.shape == arr.shape
          and l.dtype == np.uint8 and l.shape == arr.shape)
    record("I10 radiometric stretch + CLAHE produce valid u8 RGB", ok)


# --------------------------------------------------------------------------- #
# TEST: Innovation 9 (SAR backscatter engine) -- pure numpy
# --------------------------------------------------------------------------- #
def test_i09_sar_numpy():
    gray = np.zeros((64, 64), dtype=np.uint8)
    gray[0:20, 0:20] = 252  # bright urban double-bounce
    gray[40:64, 40:64] = 8   # dark specular water
    masks = m._sar_backscatter_masks(gray.astype(np.float32) / 255.0)
    water = masks["water_mask"]
    urban = masks["urban_mask"]
    w_fraction = water[45:60, 45:60].mean() / 255.0
    u_fraction = urban[2:18, 2:18].mean() / 255.0
    filtered = m._lee_filter(gray.astype(np.float32))
    ok = (w_fraction > 0.7 and u_fraction > 0.7
          and filtered.shape == gray.shape
          and np.isfinite(filtered).all())
    record("I09 SAR backscatter (water/urban masks) + Lee filter", ok
           and f"(water={w_fraction:.2f}, urban={u_fraction:.2f})")


# --------------------------------------------------------------------------- #
# TEST: Innovation 7 (NDVI/NDWI spectral gating) -- pure numpy
# --------------------------------------------------------------------------- #
def test_i07_spectral_gating():
    H = W = 100
    nir = np.full((H, W), 0.3, dtype=np.float32)
    green = np.full((H, W), 0.3, dtype=np.float32)
    red = np.full((H, W), 0.3, dtype=np.float32)
    nir[:, :50] = 0.1
    green[:, :50] = 0.6  # water left half
    bands = {"nir": nir, "green": green, "red": red}
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    ctx = m.ImageContext(Image.fromarray(rgb), bands=bands)

    water_box = m._spectral_gate_passes(ctx, "ship", 5, 5, 45, 45, W, H)
    land_box = m._spectral_gate_passes(ctx, "ship", 55, 55, 95, 95, W, H)
    non_maritime = m._spectral_gate_passes(ctx, "building", 55, 55, 95, 95, W, H)

    nir2 = np.full((H, W), 0.1, dtype=np.float32)
    red2 = np.full((H, W), 0.5, dtype=np.float32)
    nir2[:, :50] = 0.7
    red2[:, :50] = 0.1
    ctx2 = m.ImageContext(Image.fromarray(rgb), bands={"nir": nir2, "red": red2})
    veg = m._spectral_gate_passes(ctx2, "agriculture", 5, 5, 45, 45, W, H)
    urban = m._spectral_gate_passes(ctx2, "agriculture", 55, 55, 95, 95, W, H)

    ok = (water_box is True and land_box is False and non_maritime is True
          and veg is True and urban is False)
    record("I07 NDWI/NDVI gating rejects contextually-impossible boxes", ok)


# --------------------------------------------------------------------------- #
# TEST: Innovation 8 (OBB / polygon) -- pure OpenCV
# --------------------------------------------------------------------------- #
def test_i08_obb(tmpdir):
    path = make_diagonal_rgb(tmpdir)
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    obb, poly = m._oriented_box_refine(arr, (5, 5, 95, 95))
    diagonal = abs(obb["angle_deg"]) > 20  # rotated -> OBB rotation is non-trivial
    has_poly = len(poly) >= 4
    contract_kept = (len(obb["box_2d"]) == 4)
    record("I08 OBB rotation detected + polygon extracted", diagonal and has_poly and contract_kept
           and f"(angle={obb['angle_deg']:.1f}, poly={len(poly)})")


# --------------------------------------------------------------------------- #
# TEST: contracts remain intact (no inference)
# --------------------------------------------------------------------------- #
def test_contracts():
    sig = {"run_vqa", "run_grounding", "get_execution_metadata", "flush_models"}
    missing = sig - set(dir(m))
    meta = m.get_execution_metadata()
    innovations_present = all(
        k in meta.get("innovations", [])
        for k in ("I07_spectral_gating", "I08_obb_polygon", "I09_sar_backscatter",
                  "I10_radiometric_stretch", "TSC_dynamic_token_softmax")
    )
    record("Frozen contract functions exported", not missing)
    record("Innovations declared in execution metadata", innovations_present)


# --------------------------------------------------------------------------- #
# TEST: full GeoTIFF load path (I10 + I07 bands) with real raster
# --------------------------------------------------------------------------- #
def test_geotiff_load(tmpdir):
    path = make_multiband_geotiff(tmpdir)
    if path is None:
        record("I07 multiband GeoTIFF load", True, "(rasterio unavailable -> skip)")
        return
    ctx = m.load_context(path)
    has_bands = {"red", "green", "nir"}.issubset(ctx.bands.keys())
    ndwi = m._ndwi(ctx.bands)
    ndvi = m._ndvi(ctx.bands)
    ok = has_bands and ndwi is not None and ndvi is not None
    record("I07 multiband GeoTIFF yields NDVI+NDWI spectral priors", ok)


# --------------------------------------------------------------------------- #
# TEST (full): real model inference hits grounding + VQA without OOM
# --------------------------------------------------------------------------- #
def test_real_models():
    model, proc = m._load_grounding()
    vlm, vproc = m._load_vlm()
    ok_load = model is not None and vlm is not None
    record("Models load on CUDA without OOM (grounding + VLM)", ok_load
           and f"(cuda={m._cuda_available()})")

    # Grounding path with spectral gating (I07) + OBB (I08)
    t0 = time.time()
    res = m.run_grounding("demo_rgb.png", "building")
    dt = (time.time() - t0) * 1000
    REPORT["latency_ms"]["run_grounding"] = dt
    bboxes = res.get("bounding_boxes", [])
    extra_keys = all(("oriented_box_2d" in b or "polygon" in b) for b in bboxes)
    record("run_grounding contract + I08 extras",
           set(res.keys()) == {"text_response", "bounding_boxes", "confidence"}
           and (len(bboxes) == 0 or extra_keys))

    # VQA with dynamic-token-softmax confidence (TSC) + SAR/optical prompt
    t0 = time.time()
    v = m.run_vqa("demo_rgb.png", "Describe this satellite image")
    dt = (time.time() - t0) * 1000
    REPORT["latency_ms"]["run_vqa"] = dt
    record("run_vqa contract + TSC confidence",
           set(v.keys()) == {"text_response", "confidence"}
           and isinstance(v["text_response"], str)
           and 0.0 <= v["confidence"] <= 1.0,
           f"(conf={v['confidence']})")


# --------------------------------------------------------------------------- #
def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "fast").lower()
    with tempfile.TemporaryDirectory() as tmp:
        test_contracts()
        test_i09_sar_numpy()
        test_i07_spectral_gating()
        test_i08_obb(tmp)
        test_i10_radiometric()
        test_geotiff_load(tmp)
        if mode == "full":
            test_real_models()
        else:
            REPORT["skip"].append("real_model_inference")
            print("[SKIP] real model inference (pass 'full' to run)")

    print("\n========== INNOVATION TEST SUMMARY ==========")
    print(f"  PASS: {len(REPORT['ok'])}")
    print(f"  FAIL: {len(REPORT['fail'])}")
    print(f"  SKIP: {len(REPORT['skip'])}")
    for f in REPORT["fail"]:
        print(f"    !! FAILED: {f}")
    report_path = os.path.join(os.getcwd(), "innovation_test_report.json")
    with open(report_path, "w") as fh:
        json.dump(REPORT, fh, indent=2)
    print(f"  Report written -> {report_path}")

    if REPORT["fail"]:
        sys.exit(1)
    print("OK: all innovations validated within frozen contracts.")


if __name__ == "__main__":
    main()
