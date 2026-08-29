"""Explicit modality resolution for satellite/aerial rasters.

PLAN.md §4.4 requires that modality is *never* guessed silently: every
`RasterInput` must carry both the resolved modality string and a record of
which tier decided it, so the controller can put that provenance into the
execution trace (PLAN.md §4.2 `inputs[].modality`).

Precedence (highest wins):
  tier 1 "user"       - explicit caller/UI selection (the `modality=` argument
                         to `load_raster`). Always trusted, never overridden.
  tier 2 "filename" /
         "metadata"    - keyword heuristics over the filename and, for
                         GeoTIFFs, rasterio tag metadata. If both filename and
                         metadata hints fire and *disagree*, that is treated
                         as low-confidence and we fall through to "unknown"
                         rather than pick a side.
  tier 3 "band_count"  - band-count heuristic. 1-2 bands is inherently
                         ambiguous (Cartosat panchromatic vs. RISAT SAR both
                         live there) so it resolves to "unknown" unless a
                         higher tier already fired. 3 -> optical, 4 -> msi,
                         12-13 -> msi. Anything else -> unknown.

This module is intentionally free of any rasterio/PIL dependency: it only
reasons over plain Python values (a path string, an int band count, a dict of
metadata tags) so it can be unit tested in isolation from actual file IO.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

# Must match PLAN.md §4.4 exactly.
VALID_MODALITIES = ("optical", "msi", "sar", "unknown")

# Keyword heuristics used for both the filename (tier 2a) and rasterio/PIL
# metadata tag (tier 2b) checks. Order within a tuple does not matter; these
# are substring checks against a lower-cased blob of text, so keep entries
# reasonably specific to avoid false positives on unrelated words.
_SAR_KEYWORDS = (
    "sar", "risat", "grd", "_vv", "vv_", "_vh", "vh_", "vv.", "vh.",
    "sentinel-1", "sentinel1", "s1_", "_s1", "radarsat", "backscatter",
    "polsar", "hh_", "_hh",
)
_OPTICAL_KEYWORDS = (
    "optical", "rgb", "panchromatic", "_pan", "pan_", "pan.",
    "true_color", "truecolor",
)
_MSI_KEYWORDS = (
    "msi", "multispectral", "sentinel-2", "sentinel2", "s2_", "_s2",
    "cartosat_msi", "cartosat2s_msi", "bigearthnet", "ben_s2",
)

_KEYWORD_TABLE = (
    ("sar", _SAR_KEYWORDS),
    ("optical", _OPTICAL_KEYWORDS),
    ("msi", _MSI_KEYWORDS),
)


@dataclass(frozen=True)
class ModalityDecision:
    """Record of how a raster's modality was resolved.

    Attributes:
        modality: one of VALID_MODALITIES.
        mechanism: "user" | "filename" | "metadata" | "band_count" - which
            precedence tier produced `modality`.
        reason: short human-readable explanation, safe to drop straight into
            the execution trace for auditability.
    """

    modality: str
    mechanism: str
    reason: str


def _keyword_hit(text: str) -> Optional[tuple[str, str]]:
    """Scan `text` (already lower-cased) for a modality keyword.

    Returns (modality, matched_keyword) for the first match found while
    walking sar -> optical -> msi, or None if nothing matches. SAR is checked
    first because its keywords ("vv", "vh", "grd") are the most specific and
    least likely to accidentally collide with the others.
    """
    for modality, keywords in _KEYWORD_TABLE:
        for kw in keywords:
            if kw in text:
                return modality, kw
    return None


def _filename_hint(source_path: str) -> Optional[tuple[str, str]]:
    if not source_path:
        return None
    stem = os.path.basename(source_path).lower()
    return _keyword_hit(stem)


def _metadata_hint(metadata_tags: Optional[Mapping]) -> Optional[tuple[str, str]]:
    if not metadata_tags:
        return None
    try:
        blob = " ".join(f"{k}={v}" for k, v in metadata_tags.items()).lower()
    except Exception:
        return None
    return _keyword_hit(blob)


def _band_count_hint(band_count: int) -> Optional[str]:
    """Tier-3 band-count heuristic per PLAN.md §4.4.

    1-2 bands is deliberately ambiguous (Cartosat PAN and RISAT SAR both live
    there) and returns None so the caller falls back to "unknown".
    """
    if band_count == 3:
        return "optical"
    if band_count == 4:
        return "msi"
    if band_count in (12, 13):
        return "msi"
    return None


def resolve_modality(
    *,
    user_modality: Optional[str] = None,
    source_path: str = "",
    band_count: int = 0,
    metadata_tags: Optional[Mapping] = None,
) -> ModalityDecision:
    """Resolve a raster's modality using the tiered precedence in PLAN.md §4.4.

    Args:
        user_modality: explicit override supplied by the caller/UI (tier 1).
        source_path: file path, used for the filename heuristic (tier 2a).
        band_count: number of bands in the raster, used for the band-count
            heuristic (tier 3).
        metadata_tags: raster metadata (e.g. rasterio `src.tags()`), used for
            the metadata heuristic (tier 2b).

    Returns:
        ModalityDecision naming the resolved modality and which tier decided
        it. Never raises for an ambiguous input - ambiguity resolves to
        "unknown", which is treated as a legitimate, honest answer.

    Raises:
        ValueError: if `user_modality` is supplied but is not one of
            VALID_MODALITIES.
    """
    # Tier 1: explicit user selection always wins and is never second-guessed.
    if user_modality is not None:
        normalized = user_modality.strip().lower()
        if normalized not in VALID_MODALITIES:
            raise ValueError(
                f"Invalid modality override {user_modality!r}; must be one of "
                f"{VALID_MODALITIES}"
            )
        return ModalityDecision(
            modality=normalized,
            mechanism="user",
            reason=f"explicit user-supplied override: {user_modality!r}",
        )

    filename_hit = _filename_hint(source_path)
    metadata_hit = _metadata_hint(metadata_tags)

    # Tier 2: filename / metadata heuristics. If both fire and disagree, that
    # is a low-confidence signal -> resolve to unknown rather than guess.
    if filename_hit and metadata_hit:
        if filename_hit[0] == metadata_hit[0]:
            return ModalityDecision(
                modality=filename_hit[0],
                mechanism="metadata",
                reason=(
                    f"filename keyword {filename_hit[1]!r} and metadata "
                    f"keyword {metadata_hit[1]!r} agree on "
                    f"'{filename_hit[0]}'"
                ),
            )
        return ModalityDecision(
            modality="unknown",
            mechanism="metadata",
            reason=(
                f"filename hint '{filename_hit[0]}' (matched {filename_hit[1]!r}) "
                f"conflicts with metadata hint '{metadata_hit[0]}' "
                f"(matched {metadata_hit[1]!r}); resolving to unknown rather "
                f"than guess"
            ),
        )
    if filename_hit:
        return ModalityDecision(
            modality=filename_hit[0],
            mechanism="filename",
            reason=(
                f"filename keyword {filename_hit[1]!r} in "
                f"{os.path.basename(source_path)!r} indicates "
                f"'{filename_hit[0]}'"
            ),
        )
    if metadata_hit:
        return ModalityDecision(
            modality=metadata_hit[0],
            mechanism="metadata",
            reason=f"raster metadata tag keyword {metadata_hit[1]!r} indicates '{metadata_hit[0]}'",
        )

    # Tier 3: band-count heuristic.
    band_guess = _band_count_hint(band_count)
    if band_guess is not None:
        return ModalityDecision(
            modality=band_guess,
            mechanism="band_count",
            reason=f"band count {band_count} maps to '{band_guess}' with no other hint available",
        )

    return ModalityDecision(
        modality="unknown",
        mechanism="band_count",
        reason=(
            f"band count {band_count} is ambiguous and no filename/metadata "
            f"hint was available; resolving to unknown rather than guess"
        ),
    )
