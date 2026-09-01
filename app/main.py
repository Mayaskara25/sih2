"""
SATQueryAI

Run from the project root:

    streamlit run app/main.py
"""

from __future__ import annotations

import io
import json
import os
import tempfile

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pydeck as pdk
import streamlit as st

from PIL import Image

from rasterio.io import MemoryFile

from rasterio.warp import (
    Resampling,
    reproject,
    transform,
    transform_bounds,
)


# ============================================================================
# SATQUERY IMPORTS
# ============================================================================

from app.theme import inject_theme

from satquery.controller import run_query

from satquery.io.raster import (
    RasterInput,
    load_raster,
)

from satquery.report import (
    format_confidence_basis,
    generate_report,
)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="SATQueryAI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================================
# CONSTANTS
# ============================================================================

EXAMPLE_QUERIES = [
    ("VQA", "How many buildings are visible in this image?"),
    ("Captioning", "Describe the land-cover and major objects visible."),
    ("Grounding", "Highlight the water body referred to in the query."),
    ("Change", "What changed between these two dates?"),
    ("Fusion", "Use the optical and SAR images to analyse this area."),
]


# ============================================================================
# UI HELPERS
# ============================================================================


def workspace_title(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="workspace-title">{title}</div>
        <div class="workspace-sub">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def app_header() -> None:
    st.markdown(
        """
        <div class="workspace-title" style="font-size:38px; margin-bottom:2px;">
            SATQuery<span>AI</span>
        </div>
        <div class="workspace-sub" style="margin-bottom:18px;">
            Satellite imagery, geospatial tools and AI analysis in one workspace.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# FILE HELPERS
# ============================================================================


def get_bytes(uploaded_file) -> bytes:
    return uploaded_file.getvalue()


def save_upload(uploaded_file, suffix: str = ".tif") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name


# ============================================================================
# RASTER METADATA
# ============================================================================


def read_metadata(file_bytes: bytes) -> Dict[str, Any]:
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            return {
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "crs": str(src.crs) if src.crs else None,
                "resolution_x": abs(src.res[0]),
                "resolution_y": abs(src.res[1]),
                "dtype": ", ".join(src.dtypes),
                "driver": src.driver,
                "nodata": src.nodata,
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            }


def read_band(file_bytes: bytes, band_number: int = 1) -> Dict[str, Any]:
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            if band_number < 1 or band_number > src.count:
                raise ValueError(
                    f"Selected band {band_number} does not exist. "
                    f"This file contains {src.count} band(s)."
                )
            return {
                "array": src.read(band_number).astype(np.float32),
                "transform": src.transform,
                "crs": src.crs,
                "nodata": src.nodata,
                "profile": src.profile.copy(),
            }


# ============================================================================
# RASTER ALIGNMENT
# ============================================================================


def align_to_reference(source: Dict[str, Any], reference: Dict[str, Any]) -> np.ndarray:
    if (
        source["array"].shape == reference["array"].shape
        and source["crs"] == reference["crs"]
        and source["transform"] == reference["transform"]
    ):
        return source["array"]

    if source["crs"] is None or reference["crs"] is None:
        raise ValueError("Automatic spatial alignment requires georeferencing.")

    destination = np.full(reference["array"].shape, np.nan, dtype=np.float32)

    reproject(
        source=source["array"],
        destination=destination,
        src_transform=source["transform"],
        src_crs=source["crs"],
        dst_transform=reference["transform"],
        dst_crs=reference["crs"],
        src_nodata=source["nodata"],
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    return destination


# ============================================================================
# NORMALIZATION
# ============================================================================


def normalize_band(array: np.ndarray) -> np.ndarray:
    valid = array[np.isfinite(array)]
    if len(valid) == 0:
        return np.zeros(array.shape, dtype=np.uint8)

    low = np.percentile(valid, 2)
    high = np.percentile(valid, 98)

    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)

    normalized = np.clip((array - low) / (high - low), 0, 1)
    normalized = np.nan_to_num(normalized)
    return (normalized * 255).astype(np.uint8)


# ============================================================================
# RGB
# ============================================================================


def create_rgb(red_file, red_band, green_file, green_band, blue_file, blue_band) -> np.ndarray:
    red = read_band(red_file, red_band)
    green = read_band(green_file, green_band)
    blue = read_band(blue_file, blue_band)

    green_array = align_to_reference(green, red)
    blue_array = align_to_reference(blue, red)

    return np.dstack(
        [
            normalize_band(red["array"]),
            normalize_band(green_array),
            normalize_band(blue_array),
        ]
    )


def rgb_to_png(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


# ============================================================================
# SPECTRAL INDICES
# ============================================================================


def calculate_index(first_file, first_band, second_file, second_band) -> Tuple[np.ndarray, Dict[str, Any]]:
    first = read_band(first_file, first_band)
    second = read_band(second_file, second_band)

    second_array = align_to_reference(second, first)
    first_array = first["array"]

    denominator = first_array + second_array

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            np.abs(denominator) > 1e-8,
            (first_array - second_array) / denominator,
            np.nan,
        )

    return np.clip(result, -1, 1), first["profile"]


def ndvi_preview(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array)
    output = np.zeros((*array.shape, 3), dtype=np.uint8)
    output[array < 0] = [45, 75, 120]
    output[(array >= 0) & (array < 0.2)] = [175, 160, 95]
    output[(array >= 0.2) & (array < 0.5)] = [135, 210, 60]
    output[array >= 0.5] = [175, 255, 0]
    return output


def ndwi_preview(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array)
    output = np.zeros((*array.shape, 3), dtype=np.uint8)
    output[array <= 0] = [100, 110, 75]
    output[(array > 0) & (array < 0.3)] = [90, 180, 220]
    output[array >= 0.3] = [35, 125, 235]
    return output


def get_index_stats(array: np.ndarray) -> Optional[Dict[str, float]]:
    valid = array[np.isfinite(array)]
    if len(valid) == 0:
        return None
    return {
        "minimum": float(np.min(valid)),
        "mean": float(np.mean(valid)),
        "maximum": float(np.max(valid)),
    }


def index_to_geotiff(index_array: np.ndarray, profile: Dict[str, Any]) -> bytes:
    output_profile = profile.copy()
    output_profile.update(driver="GTiff", dtype="float32", count=1, compress="deflate")

    with MemoryFile() as memfile:
        with memfile.open(**output_profile) as dst:
            dst.write(index_array.astype(np.float32), 1)
        return memfile.read()


# ============================================================================
# GEOGRAPHIC FUNCTIONS
# ============================================================================


def pixel_to_latlon(file_bytes: bytes, x: int, y: int) -> Tuple[float, float]:
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            if src.crs is None:
                raise ValueError("This TIFF contains no geographic CRS.")

            if x < 0 or x >= src.width or y < 0 or y >= src.height:
                raise ValueError("Pixel is outside the image.")

            map_x, map_y = src.xy(int(y), int(x))
            longitude, latitude = transform(src.crs, "EPSG:4326", [map_x], [map_y])

            return float(latitude[0]), float(longitude[0])


def bbox_to_geojson(file_bytes: bytes, x1: int, y1: int, x2: int, y2: int) -> Dict[str, Any]:
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    coordinates = []

    for x, y in corners:
        latitude, longitude = pixel_to_latlon(file_bytes, x, y)
        coordinates.append([longitude, latitude])

    return {
        "type": "Feature",
        "properties": {"label": "Detected Region"},
        "geometry": {"type": "Polygon", "coordinates": [coordinates]},
    }


def get_wgs84_bounds(file_bytes: bytes) -> Optional[Tuple[float, float, float, float]]:
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            if src.crs is None:
                return None
            return transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)


def calculate_overlap(bounds1, bounds2) -> Optional[float]:
    if bounds1 is None or bounds2 is None:
        return None

    left = max(bounds1[0], bounds2[0])
    bottom = max(bounds1[1], bounds2[1])
    right = min(bounds1[2], bounds2[2])
    top = min(bounds1[3], bounds2[3])

    if right <= left or top <= bottom:
        return 0

    intersection = (right - left) * (top - bottom)
    area1 = (bounds1[2] - bounds1[0]) * (bounds1[3] - bounds1[1])
    area2 = (bounds2[2] - bounds2[0]) * (bounds2[3] - bounds2[1])
    smaller = min(area1, area2)

    if smaller == 0:
        return 0

    return intersection / smaller * 100


def validate_pair(file1: bytes, file2: bytes) -> Dict[str, Any]:
    metadata1 = read_metadata(file1)
    metadata2 = read_metadata(file2)

    overlap = calculate_overlap(get_wgs84_bounds(file1), get_wgs84_bounds(file2))

    return {
        "same_crs": metadata1["crs"] == metadata2["crs"],
        "same_dimensions": (
            metadata1["width"] == metadata2["width"]
            and metadata1["height"] == metadata2["height"]
        ),
        "same_resolution": np.allclose(
            [metadata1["resolution_x"], metadata1["resolution_y"]],
            [metadata2["resolution_x"], metadata2["resolution_y"]],
        ),
        "overlap": overlap,
    }


# ============================================================================
# BAND SELECTION
# ============================================================================


def find_default_file(names: List[str], token: str) -> int:
    for index, name in enumerate(names):
        if token.upper() in name.upper():
            return index
    return 0


def choose_band(files: Dict[str, bytes], label: str, token: str, key: str) -> Tuple[bytes, int]:
    names = list(files.keys())
    if not names:
        raise ValueError("No files uploaded.")

    default_index = find_default_file(names, token)

    file_state_key = f"{key}_file"
    band_state_key = f"{key}_band"

    if file_state_key not in st.session_state:
        st.session_state[file_state_key] = names[default_index]

    if band_state_key not in st.session_state:
        st.session_state[band_state_key] = 1

    filename = st.session_state[file_state_key]
    if filename not in names:
        filename = names[default_index]
        st.session_state[file_state_key] = filename

    band_number = int(st.session_state[band_state_key])

    st.caption(f"**{label}** — {filename} · band {band_number}")

    with st.expander(f"Change {label.lower()} file/band", expanded=False):
        filename = st.selectbox(
            f"{label} file",
            names,
            index=names.index(filename),
            key=f"{key}_file_picker",
        )

        metadata = read_metadata(files[filename])

        band_number = st.number_input(
            f"{label} band",
            min_value=1,
            max_value=metadata["bands"],
            value=min(band_number, metadata["bands"]),
            step=1,
            key=f"{key}_band_picker",
        )

        st.session_state[file_state_key] = filename
        st.session_state[band_state_key] = int(band_number)

    return files[filename], int(band_number)


# ============================================================================
# MAP
# ============================================================================


def estimate_zoom(bounds) -> int:
    left, bottom, right, top = bounds
    span = max(right - left, top - bottom)

    if span < 0.001:
        return 15
    if span < 0.01:
        return 13
    if span < 0.05:
        return 11
    if span < 0.5:
        return 9
    if span < 2:
        return 7
    return 4


def create_light_map(latitude: float, longitude: float, layers, zoom: int = 13):
    return pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(latitude=latitude, longitude=longitude, zoom=zoom),
        layers=layers,
    )


# ============================================================================
# METADATA DISPLAY
# ============================================================================


def show_metadata(metadata: Dict[str, Any]) -> None:
    workspace_title(
        "GeoTIFF <span>overview.</span>",
        "Core spatial properties extracted directly from the uploaded file.",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Width", f'{metadata["width"]} px')
    col2.metric("Height", f'{metadata["height"]} px')
    col3.metric("Bands", metadata["bands"])
    col4.metric("Format", metadata["driver"])

    col1, col2, col3 = st.columns(3)
    col1.metric("CRS", metadata["crs"] or "Unavailable")
    col2.metric("X Resolution", f'{metadata["resolution_x"]:.6f}')
    col3.metric("Y Resolution", f'{metadata["resolution_y"]:.6f}')


# ============================================================================
# PREVIEW + MAP
# ============================================================================


def show_preview_and_map(primary_file: bytes, metadata: Dict[str, Any]) -> None:
    workspace_title(
        "Preview <span>&amp; location.</span>",
        "What the file looks like, and where it sits on the map.",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="small-label">Image preview</div>', unsafe_allow_html=True)
        try:
            band = read_band(primary_file, 1)
            preview = normalize_band(band["array"])
            st.image(preview, use_container_width=True)
        except Exception as error:
            st.error(str(error))

    with col2:
        st.markdown('<div class="small-label">Region on the map</div>', unsafe_allow_html=True)
        try:
            bounds = get_wgs84_bounds(primary_file)

            if bounds is None:
                st.info("This file has no CRS, so it can't be placed on a map.")
            else:
                left, bottom, right, top = bounds
                polygon = [[left, top], [right, top], [right, bottom], [left, bottom], [left, top]]
                center_lat = (bottom + top) / 2
                center_lon = (left + right) / 2

                layer = pdk.Layer(
                    "PolygonLayer",
                    data=[{"polygon": polygon}],
                    get_polygon="polygon",
                    filled=True,
                    stroked=True,
                    get_fill_color=[186, 255, 0, 55],
                    get_line_color=[186, 255, 0, 220],
                    get_line_width=3,
                )

                st.pydeck_chart(
                    create_light_map(center_lat, center_lon, [layer], zoom=estimate_zoom(bounds)),
                    use_container_width=True,
                )
        except Exception as error:
            st.error(str(error))


# ============================================================================
# RESULT DIALOGS (POPUPS)
# ============================================================================


@st.dialog("RGB Composite", width="large")
def show_rgb_dialog(rgb: np.ndarray) -> None:
    st.image(rgb, caption="Natural colour composite", use_container_width=True)
    st.download_button(
        "Download RGB Image",
        rgb_to_png(rgb),
        "satquery_rgb.png",
        "image/png",
        key="download_rgb_dialog",
    )


@st.dialog("NDVI — Vegetation Index", width="large")
def show_ndvi_dialog(ndvi: np.ndarray, profile: Dict[str, Any]) -> None:
    st.image(ndvi_preview(ndvi), caption="NDVI vegetation map", use_container_width=True)

    stats = get_index_stats(ndvi)
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Minimum", f'{stats["minimum"]:.3f}')
        c2.metric("Mean", f'{stats["mean"]:.3f}')
        c3.metric("Maximum", f'{stats["maximum"]:.3f}')

    st.download_button(
        "Download NDVI GeoTIFF",
        index_to_geotiff(ndvi, profile),
        "ndvi.tif",
        "image/tiff",
        key="download_ndvi_dialog",
    )


@st.dialog("NDWI — Water Index", width="large")
def show_ndwi_dialog(ndwi: np.ndarray, profile: Dict[str, Any]) -> None:
    st.image(ndwi_preview(ndwi), caption="NDWI water map", use_container_width=True)

    stats = get_index_stats(ndwi)
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Minimum", f'{stats["minimum"]:.3f}')
        c2.metric("Mean", f'{stats["mean"]:.3f}')
        c3.metric("Maximum", f'{stats["maximum"]:.3f}')

    st.download_button(
        "Download NDWI GeoTIFF",
        index_to_geotiff(ndwi, profile),
        "ndwi.tif",
        "image/tiff",
        key="download_ndwi_dialog",
    )


@st.dialog("Pixel Location", width="large")
def show_pixel_dialog(latitude: float, longitude: float) -> None:
    col1, col2 = st.columns(2)
    col1.metric("Latitude", f"{latitude:.6f}")
    col2.metric("Longitude", f"{longitude:.6f}")

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=[{"latitude": latitude, "longitude": longitude}],
        get_position=["longitude", "latitude"],
        get_radius=80,
        get_fill_color=[90, 160, 240],
        pickable=True,
    )

    st.pydeck_chart(create_light_map(latitude, longitude, [layer], zoom=14), use_container_width=True)


@st.dialog("Geographic Region", width="large")
def show_geojson_dialog(geojson: Dict[str, Any]) -> None:
    polygon = geojson["geometry"]["coordinates"][0]
    longitude = np.mean([point[0] for point in polygon])
    latitude = np.mean([point[1] for point in polygon])

    layer = pdk.Layer(
        "PolygonLayer",
        data=[{"polygon": polygon}],
        get_polygon="polygon",
        filled=True,
        stroked=True,
        get_fill_color=[186, 255, 0, 70],
        get_line_color=[80, 110, 30],
        get_line_width=4,
    )

    st.pydeck_chart(create_light_map(latitude, longitude, [layer]), use_container_width=True)

    st.download_button(
        "Download GeoJSON",
        json.dumps(geojson, indent=2),
        "region.geojson",
        "application/geo+json",
        key="download_geojson_dialog",
    )


# ============================================================================
# AI INPUT INSPECTION
# ============================================================================


def resolve_image_info(path: str) -> Dict[str, Any]:
    raster: RasterInput = load_raster(path)
    decision = raster.modality_decision

    return {
        "raster": raster,
        "display_rgb": raster.display_rgb,
        "band_count": raster.band_count,
        "shape": list(raster.array.shape[:2]),
        "modality": raster.modality,
        "mechanism": decision.mechanism,
        "reason": decision.reason,
        "raw_path": path,
    }


def run_query_with_info(query: str, image_infos: List[Dict[str, Any]]) -> Any:
    paths = [info["raw_path"] for info in image_infos]
    return run_query(query, paths)


# ============================================================================
# AI RESULT
# ============================================================================


def render_confidence(result: Dict[str, Any]) -> str:
    confidence = result.get("confidence")
    basis = result.get("confidence_basis", "")

    if confidence is not None and basis:
        return format_confidence_basis(basis, confidence)
    if confidence is not None:
        return f"{confidence:.2f}"
    return "N/A"


def render_result_panel(result: Dict[str, Any]) -> None:
    status = result.get("status", "success")

    if status in ("validation_failed", "error"):
        error_message = result.get("errors") or result.get("error") or "Unknown failure"
        st.error(f"**{status.replace('_', ' ').title()}:** {error_message}")
        return

    text = result.get("text_response", "")
    if text:
        st.subheader("Answer")
        st.write(text)

    confidence_line = render_confidence(result)

    if confidence_line != "N/A":
        st.subheader("Confidence")

        basis = result.get("confidence_basis", "")
        basis_labels = {
            "stub": "placeholder — specialist not yet implemented",
            "heuristic": "heuristic estimate",
            "calibrated": "calibrated probability",
            "model_logprob": "model log-probability",
        }
        basis_text = basis_labels.get(basis, basis)

        st.markdown(f"**{confidence_line}**")
        st.caption(f"Confidence basis: {basis_text}")


# ============================================================================
# ARTIFACTS
# ============================================================================


def render_artifacts(artifacts: Dict[str, Any]) -> None:
    overlay_path = artifacts.get("overlay")
    mask_path = artifacts.get("mask")

    if overlay_path and os.path.isfile(overlay_path):
        st.subheader("Visual Evidence")
        st.image(Image.open(overlay_path), caption="Overlay", use_container_width=True)

    if mask_path and os.path.isfile(mask_path):
        st.image(Image.open(mask_path), caption="Mask / Agreement Map", use_container_width=True)


# ============================================================================
# VALIDATION
# ============================================================================


def render_validation(validation: Dict[str, Any]) -> None:
    for error in validation.get("errors", []):
        st.error(f"Validation error: {error}")

    for warning in validation.get("warnings", []):
        st.warning(f"Validation warning: {warning}")


# ============================================================================
# EXECUTION SUMMARY
# ============================================================================


def render_execution_summary(trace: Dict[str, Any]) -> None:
    st.subheader("Execution Summary")

    routing = trace.get("routing", {})
    models_used = trace.get("models_used", [])
    timings = trace.get("timings_ms", {})

    st.caption(f"Run {trace.get('run_id', '')} · {trace.get('timestamp', '')}")

    alternatives = routing.get("alternatives_considered", [])

    st.markdown(f"**Task selected:** {trace.get('task_selected', '')}")
    st.markdown(f"**Routing mechanism:** {routing.get('mechanism', '')}")
    st.markdown(f"**Matched rule/exemplar:** {routing.get('matched', '')}")
    st.markdown(f"**Score:** {routing.get('score', 0):.3f}")
    st.markdown(f"**Alternatives considered:** {', '.join(alternatives) or '—'}")

    if models_used:
        st.markdown("**Models / Tools Used**")

        rows = [
            {
                "Role": model.get("role", ""),
                "Name": model.get("name", ""),
                "Revision": model.get("revision", ""),
                "Precision": model.get("precision", ""),
                "Device": model.get("device", ""),
            }
            for model in models_used
        ]

        st.table(rows)

    if timings:
        st.markdown("**Timings (ms)**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Routing", timings.get("routing", 0))
        c2.metric("Validation", timings.get("validation", 0))
        c3.metric("Inference", timings.get("inference", 0))
        c4.metric("Total", timings.get("total", 0))


# ============================================================================
# AI IMAGE INPUTS
# ============================================================================


def render_image_info_panel(info: Dict[str, Any], index: int) -> None:
    st.caption(
        f"**Image {index + 1}** — {info['band_count']}-band {info['modality']} "
        f"({info['shape'][0]} × {info['shape'][1]})"
    )


def build_ai_inputs(uploaded_files, context: str) -> List[Dict[str, Any]]:
    image_infos = []

    for index, uploaded in enumerate(uploaded_files):
        suffix = Path(uploaded.name).suffix.lower() or ".tif"
        saved_path = save_upload(uploaded, suffix=suffix)

        try:
            with st.spinner(f"Analysing image {index + 1}…"):
                info = resolve_image_info(saved_path)
        except Exception as error:
            st.error(f"Could not inspect {uploaded.name}: {error}")
            continue

        image_infos.append(info)
        render_image_info_panel(info, index)

        with st.expander(f"Preview image {index + 1}", expanded=False):
            st.image(
                info["display_rgb"],
                caption=f"{info['modality']} · {info['band_count']} bands",
                use_container_width=True,
            )

    return image_infos


# ============================================================================
# SINGLE IMAGE WORKSPACE
# ============================================================================


def single_image_workspace(uploads) -> None:
    workspace_title(
        "Start your <span>analysis.</span>",
        "Upload one GeoTIFF or multiple spectral-band TIFF files.",
    )

    if not uploads:
        st.info("👆 Upload one or more .tif / .tiff files to get started.")
        return

    files = {uploaded.name: get_bytes(uploaded) for uploaded in uploads}
    names = list(files.keys())
    default_index = find_default_file(names, "B04")

    if "primary_name" not in st.session_state or st.session_state["primary_name"] not in names:
        st.session_state["primary_name"] = names[default_index]

    st.caption(f"**Main image:** {st.session_state['primary_name']} — used for overview, map and pixel lookups")

    if len(names) > 1:
        with st.expander("Change the main image"):
            primary_name = st.selectbox(
                "Main image",
                names,
                index=names.index(st.session_state["primary_name"]),
                label_visibility="collapsed",
                key="primary_image_picker",
            )
            st.session_state["primary_name"] = primary_name

    primary_name = st.session_state["primary_name"]
    primary_file = files[primary_name]
    metadata = read_metadata(primary_file)

    show_metadata(metadata)
    show_preview_and_map(primary_file, metadata)

    workspace_title(
        "Explore <span>your data.</span>",
        "Generate composites, spectral indices and geographic outputs.",
    )

    rgb_tab, ndvi_tab, ndwi_tab, coordinates_tab, geojson_tab = st.tabs(
        ["RGB Composite", "NDVI", "NDWI", "Coordinates", "GeoJSON"]
    )

    # ------------------------------------------------------------------
    # RGB
    # ------------------------------------------------------------------

    with rgb_tab:
        st.caption("Combine three bands into a natural-colour image.")

        col1, col2, col3 = st.columns(3)
        with col1:
            red = choose_band(files, "Red", "B04", "rgb_red")
        with col2:
            green = choose_band(files, "Green", "B03", "rgb_green")
        with col3:
            blue = choose_band(files, "Blue", "B02", "rgb_blue")

        if st.button("Show RGB Composite", key="generate_rgb", type="primary"):
            try:
                rgb = create_rgb(red[0], red[1], green[0], green[1], blue[0], blue[1])
                show_rgb_dialog(rgb)
            except Exception as error:
                st.error(str(error))

    # ------------------------------------------------------------------
    # NDVI
    # ------------------------------------------------------------------

    with ndvi_tab:
        st.caption("Highlights vegetation using (NIR - Red) / (NIR + Red).")

        col1, col2 = st.columns(2)
        with col1:
            nir = choose_band(files, "NIR", "B08", "ndvi_nir")
        with col2:
            red = choose_band(files, "Red", "B04", "ndvi_red")

        if st.button("Show NDVI", key="calculate_ndvi", type="primary"):
            try:
                ndvi, profile = calculate_index(nir[0], nir[1], red[0], red[1])
                show_ndvi_dialog(ndvi, profile)
            except Exception as error:
                st.error(str(error))

    # ------------------------------------------------------------------
    # NDWI
    # ------------------------------------------------------------------

    with ndwi_tab:
        st.caption("Highlights water using (Green - NIR) / (Green + NIR).")

        col1, col2 = st.columns(2)
        with col1:
            green = choose_band(files, "Green", "B03", "ndwi_green")
        with col2:
            nir = choose_band(files, "NIR", "B08", "ndwi_nir")

        if st.button("Show NDWI", key="calculate_ndwi", type="primary"):
            try:
                ndwi, profile = calculate_index(green[0], green[1], nir[0], nir[1])
                show_ndwi_dialog(ndwi, profile)
            except Exception as error:
                st.error(str(error))

    # ------------------------------------------------------------------
    # COORDINATES
    # ------------------------------------------------------------------

    with coordinates_tab:
        st.caption("Convert an image pixel into latitude and longitude.")

        col1, col2 = st.columns(2)
        with col1:
            x = st.number_input("Pixel X", min_value=0, max_value=max(metadata["width"] - 1, 0), value=0, key="pixel_x")
        with col2:
            y = st.number_input("Pixel Y", min_value=0, max_value=max(metadata["height"] - 1, 0), value=0, key="pixel_y")

        if st.button("Locate Pixel", key="locate_pixel", type="primary"):
            try:
                latitude, longitude = pixel_to_latlon(primary_file, x, y)
                show_pixel_dialog(latitude, longitude)
            except Exception as error:
                st.error(str(error))

    # ------------------------------------------------------------------
    # GEOJSON
    # ------------------------------------------------------------------

    with geojson_tab:
        st.caption("Convert a pixel rectangle into a geographic GeoJSON polygon.")

        col1, col2, col3, col4 = st.columns(4)
        x1 = col1.number_input("Top-left X", 0, max(metadata["width"] - 1, 0), 0, key="geo_x1")
        y1 = col2.number_input("Top-left Y", 0, max(metadata["height"] - 1, 0), 0, key="geo_y1")
        x2 = col3.number_input("Bottom-right X", 0, max(metadata["width"] - 1, 0), max(metadata["width"] - 1, 0), key="geo_x2")
        y2 = col4.number_input("Bottom-right Y", 0, max(metadata["height"] - 1, 0), max(metadata["height"] - 1, 0), key="geo_y2")

        if st.button("Show Geographic Region", key="generate_geojson", type="primary"):
            try:
                geojson = bbox_to_geojson(primary_file, x1, y1, x2, y2)
                show_geojson_dialog(geojson)
            except Exception as error:
                st.error(str(error))


# ============================================================================
# PAIR VALIDATION
# ============================================================================


def pair_validation_panel(first, second) -> None:
    if not first or not second:
        st.info("Upload both GeoTIFFs to run the compatibility check.")
        return

    try:
        result = validate_pair(get_bytes(first), get_bytes(second))
    except Exception as error:
        st.error(f"Could not validate the pair: {error}")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CRS Match", "YES" if result["same_crs"] else "NO")
    col2.metric("Dimensions", "MATCH" if result["same_dimensions"] else "DIFFER")
    col3.metric("Resolution", "MATCH" if result["same_resolution"] else "DIFFER")

    overlap = result["overlap"]
    col4.metric("Overlap", f"{overlap:.1f}%" if overlap is not None else "N/A")

    if overlap is not None and overlap > 80:
        st.success("Strong geographic compatibility detected.")
    elif overlap is not None and overlap > 0:
        st.warning("The images only partially overlap.")
    elif overlap == 0:
        st.error("These images do not overlap geographically.")


def pair_mode(title: str, first_label: str, second_label: str, key: str):
    workspace_title(title, "Upload two observations and verify their spatial compatibility.")

    col1, col2 = st.columns(2)
    with col1:
        first = st.file_uploader(first_label, type=["tif", "tiff"], key=f"{key}_first")
    with col2:
        second = st.file_uploader(second_label, type=["tif", "tiff"], key=f"{key}_second")

    pair_validation_panel(first, second)

    if first and second:
        st.markdown("### Pair previews")
        c1, c2 = st.columns(2)

        for col, uploaded, label in [(c1, first, first_label), (c2, second, second_label)]:
            with col:
                try:
                    path = save_upload(uploaded, suffix=".tif")
                    info = resolve_image_info(path)
                    st.image(
                        info["display_rgb"],
                        caption=f"{label} · {info['modality']}",
                        use_container_width=True,
                    )
                except Exception as error:
                    st.error(str(error))

    return first, second


# ============================================================================
# AI WORKSPACE
# ============================================================================


def ai_analysis_workspace(uploaded_files, context: str) -> None:
    if not uploaded_files:
        return

    workspace_title(
        "AI <span>analysis.</span>",
        "Describe what you want to understand — SATQueryAI routes your request "
        "to the appropriate specialist automatically.",
    )

    image_infos = build_ai_inputs(uploaded_files, context)

    if not image_infos:
        return

    query_key = f"{context}_main_ai_query"
    trace_key = f"{context}_last_ai_trace"
    query_state_key = f"{context}_last_ai_query"

    query = st.text_area(
        "Natural language query",
        placeholder="e.g. How many buildings are visible?",
        height=100,
        key=query_key,
    )

    st.markdown("**Example queries**")
    example_cols = st.columns(len(EXAMPLE_QUERIES))

    for col, (label, example) in zip(example_cols, EXAMPLE_QUERIES):
        with col:
            if st.button(label, key=f"{context}_ai_example_{label}", use_container_width=True):
                st.session_state[query_key] = example
                st.rerun()

    if st.button(
        "Run AI Analysis",
        type="primary",
        disabled=not query.strip(),
        key=f"{context}_run_ai_analysis",
    ):
        with st.spinner("Running SATQueryAI analysis — model loading may take a while on first run…"):
            try:
                trace = run_query_with_info(query, image_infos)
                st.session_state[trace_key] = trace
                st.session_state[query_state_key] = query
            except Exception as error:
                st.error(f"Analysis crashed: {error}")
                st.info("Check the console and runs/ for the execution trace.")
                return

    trace = st.session_state.get(trace_key)
    if not trace:
        return

    st.divider()

    workspace_title(
        "AI <span>result.</span>",
        "Model output, visual evidence, validation and execution details.",
    )

    previous_query = st.session_state.get(query_state_key, query)
    st.markdown(f"**You asked:** {previous_query}")

    validation = trace.get("validation", {})
    result = trace.get("result", {})
    artifacts = trace.get("artifacts", {})

    st.subheader("Input Images")
    image_cols = st.columns(len(image_infos))

    for col, info in zip(image_cols, image_infos):
        with col:
            st.image(
                info["display_rgb"],
                caption=f"Image · {info['modality']} · {info['band_count']} bands",
                use_container_width=True,
            )

    render_result_panel(result)
    render_artifacts(artifacts)
    render_validation(validation)
    render_execution_summary(trace)

    st.divider()
    st.subheader("Download Report")

    try:
        report_dir = tempfile.mkdtemp()
        report_path = os.path.join(report_dir, "satquery_report.html")

        generate_report(trace, report_path)

        with open(report_path, "rb") as report_file:
            st.download_button(
                "Download HTML Report",
                data=report_file.read(),
                file_name=f"satquery_{trace.get('run_id', 'report')}.html",
                mime="text/html",
                key=f"{context}_download_ai_report",
            )
    except Exception as error:
        st.error(f"Could not generate report: {error}")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    inject_theme()
    app_header()

    tab_single, tab_temporal, tab_sar = st.tabs(
        ["Single Image", "Bi-Temporal Pair", "Optical + SAR"]
    )

    with tab_single:
        workspace_title(
            "Satellite <span>imagery.</span>",
            "Upload one multispectral GeoTIFF or several spectral-band TIFF files.",
        )

        uploads = st.file_uploader(
            "Upload satellite imagery",
            type=["tif", "tiff"],
            accept_multiple_files=True,
            help=(
                "You can upload one multiband GeoTIFF or individual files such as "
                "B02.tif, B03.tif, B04.tif and B08.tif."
            ),
            key="single_workspace_upload",
        )

        if uploads:
            single_image_workspace(uploads)
            st.divider()
            ai_analysis_workspace(uploads[:2], "single")
        else:
            st.info(
                "Upload satellite imagery to activate metadata, maps, "
                "spectral analysis and AI tools."
            )

    with tab_temporal:
        first, second = pair_mode(
            "Compare two <span>observations.</span>",
            "Earlier GeoTIFF",
            "Later GeoTIFF",
            "temporal",
        )

        if first and second:
            st.divider()
            ai_analysis_workspace([first, second], "temporal")

    with tab_sar:
        first, second = pair_mode(
            "Combine sensor <span>perspectives.</span>",
            "Optical GeoTIFF",
            "SAR GeoTIFF",
            "optical_sar",
        )

        if first and second:
            st.divider()
            ai_analysis_workspace([first, second], "sar")


if __name__ == "__main__":
    main()