"""
Image IO — the ONLY place in the repo where a raster is ever opened (PLAN.md §4.4).

OWNED BY W0 AND FROZEN (PLAN.md §5.1).
"""

from satquery.io.modality import ModalityDecision, resolve_modality
from satquery.io.raster import RasterInput, RasterReadError, load_raster

__all__ = [
    "RasterInput",
    "RasterReadError",
    "load_raster",
    "ModalityDecision",
    "resolve_modality",
]
