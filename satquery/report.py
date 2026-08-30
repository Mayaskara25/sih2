"""Self-contained HTML report generator (PLAN.md W7 §5).

Produces a standalone HTML file containing query, inputs, answer, confidence,
visual evidence, and the FULL execution trace. No external dependencies —
images are embedded as base64 data URIs. Keeps the dependency surface zero
(PDF toolchains are heavy and fragile; §5.6 bans editing pyproject.toml).
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["generate_report", "format_confidence_basis"]


def _file_to_data_uri(path: str, mime: str = "image/png") -> Optional[str]:
    """Read an image file from disk and return a base64 data URI."""
    p = Path(path)
    if not p.is_file():
        return None
    raw = p.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def format_confidence_basis(basis: str, confidence: float) -> str:
    """Human-readable label for a confidence value and its basis."""
    labels = {
        "stub": "Placeholder — specialist not yet implemented",
        "heuristic": "Heuristic estimate",
        "calibrated": "Calibrated probability",
        "model_logprob": "Model log-probability",
    }
    label = labels.get(basis, basis)
    return f"{confidence:.2f} ({label})"


def generate_report(trace: Dict[str, Any], output_path: str) -> str:
    """Generate a self-contained HTML report from an execution trace.

    The report includes:
    - Query, task, timestamp, run ID
    - Input images with modality and modality-decision mechanism
    - Answer with confidence and its basis (never presenting a stub as a real probability)
    - Visual evidence (artifacts)
    - Full execution trace (routing, models_used, timings, validation)
    - Embedded input images as base64 data URIs

    Returns the written file path.
    """
    inputs = trace.get("inputs", [])
    result = trace.get("result", {})
    artifacts = trace.get("artifacts", {})
    validation = trace.get("validation", {})
    routing = trace.get("routing", {})
    models_used = trace.get("models_used", [])
    timings = trace.get("timings_ms", {})

    status = result.get("status", "success")
    text_response = result.get("text_response", "")
    confidence = result.get("confidence")
    confidence_basis = result.get("confidence_basis", "")

    task_label = trace.get("task_selected", "unknown")
    is_failed = status in ("validation_failed", "error")

    # Embed input images
    input_images_html = ""
    for i, inp in enumerate(inputs):
        data_uri = _file_to_data_uri(inp["path"])
        img_tag = (
            f'<img src="{data_uri}" style="max-width:100%;border-radius:6px;'
            f'border:1px solid #ddd;margin-top:4px;" />'
            if data_uri
            else f'<div style="color:#666;font-style:italic;">Image not available: '
            f'{html.escape(inp["path"])}</div>'
        )
        modality = inp.get("modality", "unknown")
        bands = inp.get("bands", "?")
        shape = inp.get("shape", ["?", "?"])
        fmt = inp.get("format", "?")
        crs = inp.get("crs") or "None"
        checks = "PASSED" if inp.get("checks_passed") else "FAILED"
        checks_color = "#2e7d32" if inp.get("checks_passed") else "#c62828"
        input_images_html += f"""
        <div style="margin-bottom:20px;padding:14px;background:#fafafa;border-radius:8px;border:1px solid #e0e0e0;">
          <h3 style="margin:0 0 8px;color:#37474f;">Input {i+1}: {html.escape(str(Path(inp['path']).name))}</h3>
          <p style="margin:2px 0;font-size:0.9em;color:#546e7a;">
            Modality: <b>{html.escape(modality)}</b> &middot; Bands: {bands} &middot;
            Shape: {shape[0]}&times;{shape[1]} &middot; Format: {html.escape(fmt)} &middot;
            CRS: {html.escape(crs)} &middot; Checks: <span style="color:{checks_color};font-weight:600;">{checks}</span>
          </p>
          {img_tag}
        </div>"""

    # Artifact images
    artifact_images = ""
    overlay_path = artifacts.get("overlay")
    if overlay_path:
        data_uri = _file_to_data_uri(overlay_path)
        if data_uri:
            artifact_images += f"""
            <div style="margin-bottom:12px;">
              <h4 style="margin:0 0 4px;">Overlay</h4>
              <img src="{data_uri}" style="max-width:100%;border-radius:6px;border:1px solid #ddd;" />
            </div>"""

    mask_path = artifacts.get("mask")
    if mask_path:
        data_uri = _file_to_data_uri(mask_path)
        if data_uri:
            artifact_images += f"""
            <div style="margin-bottom:12px;">
              <h4 style="margin:0 0 4px;">Mask / Agreement Map</h4>
              <img src="{data_uri}" style="max-width:100%;border-radius:6px;border:1px solid #ddd;" />
            </div>"""

    # Execution trace as JSON
    trace_json = html.escape(json.dumps(trace, indent=2, ensure_ascii=False))

    # Confidence line
    confidence_line = ""
    if confidence is not None and confidence_basis:
        confidence_line = format_confidence_basis(confidence_basis, confidence)
    elif confidence is not None:
        confidence_line = f"{confidence:.2f}"

    # Status banner
    status_banner = ""
    if is_failed:
        status_msg = result.get("errors") or result.get("error") or "Unknown failure"
        status_color = "#c62828" if status == "error" else "#e65100"
        status_banner = f"""
        <div style="background:{status_color};color:white;padding:10px 14px;
                    border-radius:6px;margin-bottom:14px;">
          <b>Status: {html.escape(status)}</b>
          <p style="margin:4px 0 0;">{html.escape(str(status_msg))}</p>
        </div>"""

    # Validation warnings
    warnings_html = ""
    warnings = validation.get("warnings", [])
    if warnings:
        warnings_html = "<ul>" + "".join(
            f"<li>{html.escape(w)}</li>" for w in warnings
        ) + "</ul>"

    # Models table
    models_html = ""
    for m in models_used:
        models_html += f"""
        <tr>
          <td>{html.escape(m.get('role', ''))}</td>
          <td>{html.escape(m.get('name', ''))}</td>
          <td>{html.escape(m.get('revision', ''))}</td>
          <td>{html.escape(m.get('precision', ''))}</td>
          <td>{html.escape(m.get('device', ''))}</td>
        </tr>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>SatQuery Report — {html.escape(trace.get('run_id', 'unknown'))}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           margin: 0; padding: 24px; background: #f5f5f5; color: #212121; }}
    .container {{ max-width: 960px; margin: 0 auto; background: white; padding: 32px;
                  border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }}
    h2 {{ color: #37474f; margin-top: 28px; border-bottom: 1px solid #e0e0e0; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px; }}
    th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 0.9em; }}
    th {{ background: #eceff1; color: #546e7a; }}
    .meta {{ color: #546e7a; font-size: 0.85em; }}
    pre {{ background: #263238; color: #eeffff; padding: 16px; border-radius: 8px;
           overflow-x: auto; font-size: 0.82em; line-height: 1.45; }}
    code {{ font-family: 'SFMono-Regular', Consolas, monospace; }}
    @media print {{
      body {{ background: white; padding: 0; }}
      .container {{ box-shadow: none; padding: 16px; }}
      pre {{ white-space: pre-wrap; word-break: break-word; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>SatQuery Execution Report</h1>
    <p class="meta">
      Run: <code>{html.escape(trace.get('run_id', ''))}</code> &middot;
      {html.escape(trace.get('timestamp', ''))}
    </p>
    {status_banner}

    <h2>Query</h2>
    <p style="font-size:1.1em;"><b>{html.escape(trace.get('query', ''))}</b></p>
    <p class="meta">Task selected: <b>{html.escape(task_label)}</b> &middot;
      Routing: {html.escape(routing.get('mechanism', ''))} matched
      &laquo;{html.escape(routing.get('matched', ''))}&raquo;
      (score {routing.get('score', 0):.3f})</p>

    <h2>Input Images</h2>
    {input_images_html if input_images_html else '<p class="meta">No input images.</p>'}

    <h2>Answer</h2>
    {f'<p style="font-size:1.05em;"><b>{html.escape(text_response)}</b></p>' if text_response else ''}
    {f'<p class="meta">Confidence: <b>{html.escape(confidence_line)}</b></p>' if confidence_line else ''}
    {artifact_images}

    <h2>Validation</h2>
    <p>Passed: <b>{'Yes' if validation.get('passed') else 'No'}</b></p>
    {f'<h4>Warnings</h4>{warnings_html}' if warnings_html else ''}

    <h2>Models Used</h2>
    <table>
      <thead><tr><th>Role</th><th>Name</th><th>Revision</th><th>Precision</th><th>Device</th></tr></thead>
      <tbody>{models_html if models_html else '<tr><td colspan="5" class="meta">No models loaded (stub era).</td></tr>'}</tbody>
    </table>

    <h2>Execution Summary</h2>
    <table>
      <tbody>
        <tr><th>Run ID</th><td>{html.escape(trace.get('run_id', ''))}</td></tr>
        <tr><th>Timestamp</th><td>{html.escape(trace.get('timestamp', ''))}</td></tr>
        <tr><th>Task Selected</th><td>{html.escape(task_label)}</td></tr>
        <tr><th>Routing Mechanism</th><td>{html.escape(routing.get('mechanism', ''))}</td></tr>
        <tr><th>Routing Match</th><td>{html.escape(routing.get('matched', ''))}</td></tr>
        <tr><th>Routing Score</th><td>{routing.get('score', 0):.3f}</td></tr>
        <tr><th>Alternatives Considered</th><td>{html.escape(', '.join(routing.get('alternatives_considered', [])))}</td></tr>
        <tr><th>Timing (ms)</th><td>routing={timings.get('routing', 0)} &middot;
          validation={timings.get('validation', 0)} &middot;
          inference={timings.get('inference', 0)} &middot;
          total={timings.get('total', 0)}</td></tr>
      </tbody>
    </table>

    <h2>Full Execution Trace</h2>
    <pre><code>{trace_json}</code></pre>
  </div>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html_content, encoding="utf-8")
    return output_path
