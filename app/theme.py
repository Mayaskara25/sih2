"""SatQuery visual theme (PLAN.md W10, Item 4).

Ported from ``old_files/satquery-geospatial/app.py`` in two passes, per the
W10 brief:

  1. Version-stable rules — ``:root`` custom properties, ``.stApp``,
     ``.block-container``, ``h1``/``h2``/``h3``, and the source app's own
     ``.hero`` / ``.feature`` / ``.workspace-title`` / ``.soft-card`` classes.
     None of these depend on Streamlit's internal ``data-testid`` markup, so
     they are not expected to break across Streamlit releases.
  2. Exactly four ``[data-testid=...]`` groups — sidebar, file uploader,
     metric values, and alert/error text — kept because they matter for
     legibility. The remaining ~50 testid selectors in the source file were
     dropped rather than ported speculatively (they are internal to
     Streamlit 1.62.0's own build and were not something this pass could
     confirm still bind); see docs/status/W10.md for what was actually
     confirmed to render.

Two exports, one call site: ``app/main.py`` imports ``inject_theme`` and
calls it once near the top of ``main()``. Deleting that one line and this
file's import reverts the whole restyle.

Feature-card copy is NOT carried over from the source file — it advertised
capabilities ("Multispectral visualisation", "GeoTIFF intelligence built
in", "Geographic precision") the app didn't have. Rewritten below to
describe what SatQuery actually does.
"""

from __future__ import annotations

import streamlit as st

THEME_CSS = """
<style>

:root {
    --bg: #070b0c;
    --surface: #0d1314;
    --surface-2: #11191a;
    --border: #293233;
    --lime: #baff00;
    --lime-soft: #d4ff65;
    --text: #f4f7f3;
    --muted: #899493;
}

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 20% 0%, rgba(186,255,0,0.045), transparent 24%),
        radial-gradient(circle at 80% 20%, rgba(186,255,0,0.025), transparent 22%),
        linear-gradient(180deg, #070b0c 0%, #090d0e 100%);
    color: var(--text);
}

.block-container {
    max-width: 1380px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}

h1, h2, h3 {
    color: #f7faf5 !important;
    letter-spacing: -0.5px;
}

.hero {
    padding: 42px 44px;
    border-radius: 22px;
    border: 1px solid #242c2d;
    background:
        radial-gradient(circle at 85% 15%, rgba(186,255,0,0.09), transparent 28%),
        linear-gradient(135deg, #0d1314 0%, #090e0f 100%);
    margin-bottom: 26px;
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -90px;
    top: -100px;
    border: 1px solid rgba(186,255,0,0.12);
    border-radius: 50%;
}

.hero::before {
    content: "";
    position: absolute;
    width: 140px;
    height: 140px;
    right: 40px;
    top: 20px;
    border: 1px solid rgba(186,255,0,0.08);
    border-radius: 50%;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #baff00;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

.eyebrow::before {
    content: "";
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #baff00;
    box-shadow: 0 0 10px 2px rgba(186,255,0,0.6);
}

.hero-title {
    color: #f5f7f3;
    font-size: 46px;
    line-height: 1.08;
    font-weight: 800;
    max-width: 720px;
    letter-spacing: -1.5px;
}

.hero-title span {
    color: #baff00;
}

.hero-text {
    max-width: 640px;
    color: #929c9b;
    font-size: 15px;
    line-height: 1.7;
    margin-top: 16px;
}

.feature-row {
    display: grid;
    grid-template-columns: 1fr 1.15fr 1fr;
    gap: 14px;
    margin-bottom: 30px;
}

.feature {
    min-height: 170px;
    padding: 24px;
    background: #0e1415;
    border: 1px solid #2a3233;
    border-radius: 18px;
    transition: border-color 0.15s ease, transform 0.15s ease;
}

.feature:hover {
    border-color: #3a4546;
}

.feature.highlight {
    background: #baff00;
    border-color: #baff00;
    transform: translateY(-7px);
}

.feature.highlight:hover {
    transform: translateY(-9px);
}

.feature-number {
    font-size: 15px;
    font-weight: 800;
    color: #baff00;
    font-variant-numeric: tabular-nums;
}

.feature.highlight .feature-number,
.feature.highlight .feature-title,
.feature.highlight .feature-text {
    color: #111511;
}

.feature-title {
    margin-top: 18px;
    color: white;
    font-size: 17px;
    line-height: 1.35;
    font-weight: 750;
}

.feature-text {
    margin-top: 8px;
    color: #788382;
    font-size: 12.5px;
    line-height: 1.55;
}

.workspace-title {
    font-size: 28px;
    font-weight: 800;
    margin: 18px 0 4px 0;
    letter-spacing: -0.5px;
}

.workspace-title span {
    color: #baff00;
}

.workspace-sub {
    color: #788382;
    margin-bottom: 22px;
    font-size: 14px;
}

.soft-card {
    background: #0e1415;
    border: 1px solid #283132;
    border-radius: 16px;
    padding: 20px;
    margin: 10px 0 22px 0;
}

.lime-text {
    color: #baff00;
}

.small-label {
    color: #717c7b;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.auto-pick {
    background: #0e1415;
    border: 1px solid #273031;
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 14px;
    line-height: 1.6;
}

.query-box {
    margin-top: 15px;
    padding: 13px 15px;
    background: rgba(186,255,0,0.07);
    border: 1px solid rgba(186,255,0,0.25);
    border-radius: 11px;
    color: #d9ff73;
    font-size: 13px;
    line-height: 1.5;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* --- The four data-testid groups confirmed worth keeping (legibility) --- */

[data-testid="stSidebar"] {
    background: #090e0f;
    border-right: 1px solid #202829;
}

[data-testid="stSidebar"] * {
    color: #eef4ef;
}

[data-testid="stSidebar"] textarea {
    background: #101617 !important;
    color: white !important;
    border: 1px solid #2a3435 !important;
    border-radius: 13px !important;
}

[data-testid="stSidebar"] textarea:focus {
    border-color: #baff00 !important;
    box-shadow: 0 0 0 3px rgba(186,255,0,0.12) !important;
}

[data-testid="stFileUploader"] {
    background: #0d1314;
    border: 1px dashed #384344;
    border-radius: 16px;
    padding: 12px;
    transition: border-color 0.15s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #baff00;
}

[data-testid="stMetric"] {
    background: #0e1415;
    padding: 18px;
    border: 1px solid #273031;
    border-radius: 15px;
}

[data-testid="stMetricLabel"] {
    color: #75807f !important;
}

[data-testid="stMetricValue"] {
    color: #f5f7f3 !important;
    font-weight: 750;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
"""

HERO_HTML = """
<div class="hero">
<div class="eyebrow">Satellite intelligence workspace</div>
<div class="hero-title">Understand Earth data.<br><span>Without the complexity.</span></div>
<div class="hero-text">Ask a question in plain English about one or two remote-sensing images.
SatQuery routes it to the right specialist, runs it, and shows you exactly how it got the
answer.</div>
</div>
<div class="feature-row">
<div class="feature">
<div class="feature-number">01.</div>
<div class="feature-title">Agentic<br>routing.</div>
<div class="feature-text">Your query is matched to a task automatically — no menus to configure by hand.</div>
</div>
<div class="feature highlight">
<div class="feature-number">02.</div>
<div class="feature-title">Five<br>specialists.</div>
<div class="feature-text">VQA, captioning, grounding, change detection and optical/SAR fusion, dispatched per query.</div>
</div>
<div class="feature">
<div class="feature-number">03.</div>
<div class="feature-title">Auditable<br>execution trace.</div>
<div class="feature-text">Every answer ships with the routing decision, models used, timings and confidence basis.</div>
</div>
</div>
"""


def inject_theme() -> None:
    """Apply the SatQuery dark/lime theme and render the hero banner.

    Call once, near the top of ``main()``, after ``st.set_page_config``.
    """
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(HERO_HTML, unsafe_allow_html=True)
