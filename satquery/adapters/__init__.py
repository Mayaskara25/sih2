"""
Model adapters. Mr W0 pre-declared exports here pointing at stubs; benclip.py is
W2's deliverable (PLAN.md §3.1 Track A, §4.5 band-mapping contract). W2 is the
only writer of this file and adds its exports here as part of landing benclip.

The runtime contract surface that W3/W4/W5 actually call:
  load_benclip / embed_optical / embed_sar / predict_labels
plus the band-stacking/stats helpers and the model constructors used by the
training pipeline under ``train/``.
"""

from __future__ import annotations

from satquery.adapters.benclip import (
    BEN_14_SLOTS,
    S1_BAND_ORDER,
    S2_BAND_ORDER,
    BenClipConfig,
    BenClipModel,
    build_benclip_model,
    compute_channel_stats,
    embed_optical,
    embed_sar,
    load_benclip,
    predict_labels,
    reset_default,
    stack_ben_patch,
)

__all__ = [
    "BEN_14_SLOTS",
    "S1_BAND_ORDER",
    "S2_BAND_ORDER",
    "BenClipConfig",
    "BenClipModel",
    "build_benclip_model",
    "compute_channel_stats",
    "embed_optical",
    "embed_sar",
    "load_benclip",
    "predict_labels",
    "reset_default",
    "stack_ben_patch",
]
