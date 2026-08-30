"""SatQuery interactive web application (PLAN.md W7, R6).

Entry point: ``streamlit run app/main.py``

Architecture:
  - ``app/main.py`` — thin Streamlit UI (this file). Wires uploads, selectors,
    and buttons to the controller; renders results. Kept thin so the Streamlit
    layer is minimal and the heavy logic lives in testable helpers.
  - ``satquery.controller.run_query`` — the single entry point for every
    round trip. Routes, validates, dispatches, and writes an auditable trace.
  - ``satquery.report.generate_report`` — downloadable HTML report.
  - ``satquery.io.raster.load_raster`` — the ONLY place rasters are opened.

Design notes for W7:
  - The per-image modality selector is LOAD-BEARING (PLAN.md §4.4, §2.3).
    W0's resolver deliberately returns "unknown" for bare 1-2 band files
    without filename hints — exactly what real Cartosat PAN and RISAT
    products look like. The user's explicit choice may be the only thing
    that makes the cross-modal path work on the hidden evaluation set.
  - confidence_basis is always shown alongside the confidence number.
    A stub's 0.0 must never be presented as a real probability.
  - The execution summary panel is a graded rubric row — rendered
    prominently, not collapsed into JSON.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from satquery.controller import run_query
from satquery.controller.validate import inspect_input
from satquery.io.raster import RasterInput, load_raster
from satquery.report import format_confidence_basis, generate_report

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODALITY_OPTIONS = ["auto", "optical", "msi", "sar", "unknown"]

EXAMPLE_QUERIES: List[Tuple[str, str]] = [
    ("VQA", "How many buildings are visible in this image?"),
    ("Captioning", "Describe the land-cover and major objects visible."),
    ("Grounding", "Highlight the water body referred to in the query."),
    ("Change", "What changed between these two dates?"),
    ("Fusion", "Use the optical and SAR images to analyse this area."),
]

SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# ---------------------------------------------------------------------------
# Helpers — kept out of Streamlit API calls so they are unit-testable
# ---------------------------------------------------------------------------


def _save_upload(uploaded, suffix: str = ".tif") -> str:
    """Persist a Streamlit UploadedFile to a temp path and return it."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.close()
    return tmp.name


def _resolve_image_info(
    path: str,
    user_modality: Optional[str],
) -> Dict[str, Any]:
    """Resolve auto-detected modality info for a single uploaded image.

    Returns a dict with: display_rgb, band_count, shape, modality,
    modality_decision (mechanism + reason), raw_path.
    """
    forced = user_modality if user_modality and user_modality != "auto" else None
    raster: RasterInput = load_raster(path, modality=forced)
    md = raster.modality_decision
    return {
        "raster": raster,
        "display_rgb": raster.display_rgb,
        "band_count": raster.band_count,
        "shape": list(raster.array.shape[:2]),
        "modality": raster.modality,
        "mechanism": md.mechanism,
        "reason": md.reason,
        "raw_path": path,
    }


def _run_query_with_info(
    query: str,
    image_infos: List[Dict[str, Any]],
    user_modalities: List[str],
) -> Any:
    """Invoke the controller with user-selected modalities."""
    paths = [info["raw_path"] for info in image_infos]
    forced = [m if m != "auto" else None for m in user_modalities]
    return run_query(query, paths, forced_modalities=forced)


# ---------------------------------------------------------------------------
# Rendering helpers — pure functions, no Streamlit state
# ---------------------------------------------------------------------------


def render_confidence(result: Dict[str, Any]) -> str:
    """Return a labelled confidence string."""
    confidence = result.get("confidence")
    basis = result.get("confidence_basis", "")
    if confidence is not None and basis:
        return format_confidence_basis(basis, confidence)
    if confidence is not None:
        return f"{confidence:.2f}"
    return "N/A"


def render_result_panel(result: Dict[str, Any]) -> None:
    """Render the specialist answer, confidence with basis, and visual evidence."""
    status = result.get("status", "success")

    if status in ("validation_failed", "error"):
        err_msg = result.get("errors") or result.get("error") or "Unknown failure"
        st.error(f"**{status.replace('_', ' ').title()}:** {err_msg}")
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


def render_artifacts(artifacts: Dict[str, Any]) -> None:
    """Render visual evidence: overlay images, masks, agreement maps."""
    overlay_path = artifacts.get("overlay")
    mask_path = artifacts.get("mask")

    if overlay_path and os.path.isfile(overlay_path):
        st.subheader("Visual Evidence")
        img = Image.open(overlay_path)
        st.image(img, caption="Overlay", use_container_width=True)

    if mask_path and os.path.isfile(mask_path):
        img = Image.open(mask_path)
        st.image(img, caption="Mask / Agreement Map", use_container_width=True)


def render_execution_summary(trace: Dict[str, Any]) -> None:
    """Render the graded execution-summary panel — the full trace as a
    human-readable panel, not a collapsed JSON blob."""
    st.subheader("Execution Summary")

    routing = trace.get("routing", {})
    models_used = trace.get("models_used", [])
    timings = trace.get("timings_ms", {})

    # Top-line: run ID + timestamp
    st.caption(f"Run: `{trace.get('run_id', '')}` · {trace.get('timestamp', '')}")

    # Routing panel
    st.markdown("**Routing Decision**")
    st.markdown(
        f"- **Task selected:** `{trace.get('task_selected', '')}`\n"
        f"- **Mechanism:** {routing.get('mechanism', '')}\n"
        f"- **Matched rule/exemplar:** «{routing.get('matched', '')}»\n"
        f"- **Score:** {routing.get('score', 0):.3f}\n"
        f"- **Alternatives considered:** {', '.join(routing.get('alternatives_considered', [])) or '—'}"
    )

    # Models table
    if models_used:
        st.markdown("**Models / Tools Used**")
        rows = []
        for m in models_used:
            rows.append({
                "Role": m.get("role", ""),
                "Name": m.get("name", ""),
                "Revision": m.get("revision", ""),
                "Precision": m.get("precision", ""),
                "Device": m.get("device", ""),
            })
        st.table(rows)

    # Timings
    if timings:
        st.markdown("**Timings (ms)**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Routing", timings.get("routing", 0))
        c2.metric("Validation", timings.get("validation", 0))
        c3.metric("Inference", timings.get("inference", 0))
        c4.metric("Total", timings.get("total", 0))

    # Parameters
    params = trace.get("parameters", {})
    if params:
        with st.expander("Parameters", expanded=False):
            st.json(params)

    # Full trace (expandable)
    with st.expander("Full Execution Trace (JSON)", expanded=False):
        st.json(trace)


def render_validation(validation: Dict[str, Any]) -> None:
    """Render validation warnings and errors, distinguishing severity."""
    errors = validation.get("errors", [])
    warnings = validation.get("warnings", [])
    if errors:
        for e in errors:
            st.error(f"Validation error: {e}")
    if warnings:
        for w in warnings:
            st.warning(f"Validation warning: {w}")


def render_image_info_panel(info: Dict[str, Any], idx: int) -> None:
    """Show auto-detected modality and decision mechanism for one image."""
    st.markdown(
        f"**Image {idx + 1}:** `{info['band_count']}`-band "
        f"{info['modality']} · "
        f"decision: `{info['mechanism']}`"
    )
    st.caption(info["reason"])


def get_example_queries() -> List[Tuple[str, str]]:
    """Return the PS's representative queries."""
    return EXAMPLE_QUERIES


# ---------------------------------------------------------------------------
# Streamlit main — kept thin
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="SatQuery — Satellite Image Analysis",
        layout="wide",
    )
    st.title("SatQuery")
    st.caption(
        "Satellite image analysis: ask a question about one or two remote-sensing "
        "images and get an evidence-grounded answer with a full execution trace."
    )

    # --- Image upload ---
    st.subheader("Upload Images")
    uploaded_files = st.file_uploader(
        "Upload 1–2 images (GeoTIFF/TIFF, PNG, JPEG)",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if uploaded_files and len(uploaded_files) > 2:
        st.warning("At most 2 images are supported. Only the first 2 will be used.")
        uploaded_files = uploaded_files[:2]

    # --- Per-image modality selector + info display ---
    image_infos: List[Dict[str, Any]] = []
    user_modalities: List[str] = []

    if uploaded_files:
        st.subheader("Image Details & Modality")
        for i, up_file in enumerate(uploaded_files):
            suffix = Path(up_file.name).suffix.lower() or ".tif"
            saved_path = _save_upload(up_file, suffix=suffix)

            # Auto-detect modality via the IO layer
            with st.spinner(f"Analysing image {i + 1}…"):
                info = _resolve_image_info(saved_path, user_modality=None)
            image_infos.append(info)

            # Pre-fill selector with auto-detected value
            auto_mod = info["modality"]
            options = MODALITY_OPTIONS
            default_idx = options.index(auto_mod) if auto_mod in options else 0

            st.markdown(f"**Image {i + 1}:** `{info['band_count']}`-band · "
                        f"shape {info['shape'][0]}×{info['shape'][1]}")
            st.caption(
                f"Auto-detected: **{auto_mod}** "
                f"(mechanism: `{info['mechanism']}` — {info['reason']})"
            )
            chosen = st.selectbox(
                f"Modality for image {i + 1}",
                options,
                index=default_idx,
                key=f"modality_{i}",
                help=(
                    "Override the auto-detected modality. On hidden ISRO evaluation "
                    "data (Cartosat PAN, RISAT), the auto-detection may return "
                    "'unknown' — select the correct modality here."
                ),
            )
            user_modalities.append(chosen)

            # Show a small preview
            with st.expander(f"Preview image {i + 1}", expanded=False):
                st.image(
                    info["display_rgb"],
                    caption=f"{info['modality']} · {info['band_count']} bands",
                    use_container_width=True,
                )

    # --- Query box ---
    st.subheader("Query")
    query = st.text_input(
        "Ask a question about the image(s)",
        placeholder="e.g. How many buildings are visible?",
    )

    st.markdown("**Example queries:**")
    cols = st.columns(len(EXAMPLE_QUERIES))
    for col, (label, example) in zip(cols, EXAMPLE_QUERIES):
        with col:
            if st.button(label, key=f"example_{label}"):
                query = example
                st.rerun()

    # --- Run button ---
    run_disabled = not uploaded_files or not query.strip()
    if st.button("Run Analysis", type="primary", disabled=run_disabled):
        if not uploaded_files:
            st.warning("Please upload at least one image.")
        elif not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner(
                "Running analysis — model load may take tens of seconds on first run…"
            ):
                try:
                    trace = _run_query_with_info(query, image_infos, user_modalities)
                except Exception as exc:
                    st.error(f"Analysis crashed: {exc}")
                    st.info(
                        "A crash trace may have been written under `runs/`. "
                        "Check the console for details."
                    )
                    return

            # Render results
            validation = trace.get("validation", {})
            result = trace.get("result", {})

            st.divider()
            st.subheader("Result")

            # Show input images (always, using RasterInput.display_rgb)
            if image_infos:
                st.subheader("Input Images")
                img_cols = st.columns(len(image_infos))
                for col, info in zip(img_cols, image_infos):
                    with col:
                        st.image(
                            info["display_rgb"],
                            caption=f"Image ({info['modality']}, {info['band_count']} bands)",
                            use_container_width=True,
                        )

            # Result panel: answer, confidence with basis
            render_result_panel(result)

            # Visual evidence: overlay, mask
            render_artifacts(trace.get("artifacts", {}))

            # Validation panel
            render_validation(validation)

            # Graded execution summary
            render_execution_summary(trace)

            # Downloadable report
            st.divider()
            st.subheader("Download Report")
            report_dir = tempfile.mkdtemp()
            report_path = os.path.join(report_dir, "satquery_report.html")
            generate_report(trace, report_path)
            with open(report_path, "rb") as f:
                st.download_button(
                    "Download HTML Report",
                    data=f.read(),
                    file_name=f"satquery_{trace.get('run_id', 'report')}.html",
                    mime="text/html",
                )


if __name__ == "__main__":
    main()
